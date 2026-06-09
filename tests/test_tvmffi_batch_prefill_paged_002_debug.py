import math
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.jit import gen_jit_spec
from flashinfer.jit.attention.tvm import gen_customize_batch_prefill_tvm_binding


NHD = 0
HND = 1


@lru_cache(maxsize=None)
def build_mod(head_dim):
    uri, sources = gen_customize_batch_prefill_tvm_binding(
        backend="fa2",
        uri=f"tvmffi_func_batch_prefill_fp16_h{head_dim}_paged_002_debug",
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        dtype_o=torch.float16,
        idtype=torch.int32,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        additional_tensor_names=[
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
        ],
        additional_tensor_dtypes=[
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
        ],
        additional_scalar_names=[
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ],
        additional_scalar_dtypes=["double", "double", "double", "double", "int64_t"],
        variant_name="DefaultAttention<use_custom_mask, false, false, false>",
        variant_decl="using namespace flashinfer;",
    )
    return gen_jit_spec(uri, sources).build_and_load_tvm_ffi()


def xllm_tail_args(sm_scale):
    return [
        None,
        None,
        None,
        None,
        None,
        None,
        0.0,
        sm_scale,
        1.0,
        1.0 / 10000.0,
        0,
    ]


def make_indptr(lengths, device):
    values = [0]
    for length in lengths:
        values.append(values[-1] + length)
    return torch.tensor(values, dtype=torch.int32, device=device)


def make_page_table(kv_lens, page_size, device):
    pages_per_request = [math.ceil(length / page_size) for length in kv_lens]
    kv_indptr = make_indptr(pages_per_request, device)
    total_pages = int(kv_indptr[-1].item())
    page_ids = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_indices = torch.flip(page_ids, dims=[0]).contiguous()
    last_page_len = [
        length % page_size if length % page_size != 0 else page_size for length in kv_lens
    ]
    kv_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, total_pages


def gather_kv(k_cache, v_cache, indices, start_page, end_page, last_len, layout):
    k_parts = []
    v_parts = []
    for ppos in range(start_page, end_page):
        page_idx = int(indices[ppos].item())
        valid = k_cache.shape[1] if layout == NHD else k_cache.shape[2]
        if ppos + 1 == end_page:
            valid = int(last_len)
        if layout == NHD:
            k_parts.append(k_cache[page_idx, :valid])
            v_parts.append(v_cache[page_idx, :valid])
        else:
            k_parts.append(k_cache[page_idx, :, :valid].transpose(0, 1))
            v_parts.append(v_cache[page_idx, :, :valid].transpose(0, 1))
    return torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


def ref_paged_prefill(
    q,
    k_cache,
    v_cache,
    qo_indptr,
    kv_indptr,
    kv_indices,
    kv_last_page_len,
    causal,
    sm_scale,
    layout,
):
    out = torch.empty_like(q, dtype=torch.float32)
    batch = qo_indptr.numel() - 1
    for b in range(batch):
        q0, q1 = int(qo_indptr[b].item()), int(qo_indptr[b + 1].item())
        p0, p1 = int(kv_indptr[b].item()), int(kv_indptr[b + 1].item())
        qb = q[q0:q1].float()
        kb, vb = gather_kv(
            k_cache,
            v_cache,
            kv_indices,
            p0,
            p1,
            kv_last_page_len[b].item(),
            layout,
        )
        kb = kb.float()
        vb = vb.float()
        group = qb.shape[1] // kb.shape[1]
        for h in range(qb.shape[1]):
            kh = h // group
            scores = qb[:, h] @ kb[:, kh].transpose(0, 1) * sm_scale
            if causal:
                m = qb.shape[0]
                n = kb.shape[0]
                q_idx = torch.arange(m, device=q.device)[:, None]
                kv_idx = torch.arange(n, device=q.device)[None, :]
                mask = kv_idx <= (q_idx + n - m)
                scores = scores.masked_fill(~mask, -float("inf"))
            probs = torch.softmax(scores, dim=-1)
            out[q0:q1, h] = probs @ vb[:, kh]
    return out.to(q.dtype)


def print_tensor(name, tensor, sample_count=8):
    flat = tensor.flatten()
    sample = flat[: min(sample_count, flat.numel())].detach().cpu().tolist()
    print(
        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device} "
        f"stride={tensor.stride()} contiguous={tensor.is_contiguous()} "
        f"data_ptr=0x{tensor.data_ptr():x}",
        flush=True,
    )
    print(f"{name}.sample={sample}", flush=True)


def print_batch_diffs(out, ref, qo_indptr, num_qo_heads):
    diff = (out.float() - ref.float()).abs()
    for b in range(qo_indptr.numel() - 1):
        q0, q1 = int(qo_indptr[b].item()), int(qo_indptr[b + 1].item())
        batch_diff = diff[q0:q1]
        print(
            f"batch[{b}] q_range=[{q0}, {q1}) "
            f"max_abs={float(batch_diff.max().item()):.6g} "
            f"mean_abs={float(batch_diff.mean().item()):.6g}",
            flush=True,
        )
        for h in range(num_qo_heads):
            head_diff = batch_diff[:, h]
            print(
                f"batch[{b}].head[{h}] max_abs={float(head_diff.max().item()):.6g} "
                f"mean_abs={float(head_diff.mean().item()):.6g}",
                flush=True,
            )


@pytest.mark.parametrize("case_name", ["paged_002"])
def test_tvmffi_paged_002_debug(case_name):
    head_dim = 64
    layout = HND
    causal = True
    enable_cuda_graph = False
    disable_split_kv = False
    page_size = 1
    q_lens = [129, 2, 33, 257]
    kv_lens = [130, 4, 36, 257]
    num_qo_heads = 1
    num_kv_heads = 1
    sm_scale = 1.0 / math.sqrt(head_dim)
    device = "cuda:0"

    torch.manual_seed(
        head_dim
        + layout * 13
        + int(causal) * 17
        + len(q_lens) * 19
        + num_qo_heads * 23
        + num_kv_heads * 29
        + int(enable_cuda_graph) * 31
        + int(disable_split_kv) * 37
        + page_size * 41
        + sum(q_lens) * 43
        + sum(kv_lens) * 47
    )

    print(
        f"case name={case_name} op=paged layout={layout} causal={causal} "
        f"q_lens={q_lens} kv_lens={kv_lens} heads={num_qo_heads}/{num_kv_heads} "
        f"head_dim={head_dim} page_size={page_size} sm_scale={sm_scale}",
        flush=True,
    )
    print(
        f"enable_cuda_graph={enable_cuda_graph} disable_split_kv={disable_split_kv}",
        flush=True,
    )

    mod = build_mod(head_dim)
    batch = len(q_lens)
    qo_indptr = make_indptr(q_lens, device)
    kv_indptr, kv_indices, kv_last_page_len, total_pages = make_page_table(
        kv_lens, page_size, device
    )
    qo_indptr_h = qo_indptr.cpu().contiguous()
    kv_indptr_h = kv_indptr.cpu().contiguous()
    kv_len_arr_h = torch.tensor(kv_lens, dtype=torch.int32).contiguous()
    total_num_rows = int(qo_indptr_h[-1].item())

    print(f"qo_indptr_h={qo_indptr_h.tolist()}", flush=True)
    print(f"kv_indptr_h={kv_indptr_h.tolist()}", flush=True)
    print(f"kv_len_arr_h={kv_len_arr_h.tolist()}", flush=True)
    print(f"kv_indices_h={kv_indices.cpu().tolist()}", flush=True)
    print(f"kv_last_page_len_h={kv_last_page_len.cpu().tolist()}", flush=True)
    print(f"total_pages={total_pages}", flush=True)

    q = torch.randn(total_num_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k_cache = torch.randn(
        total_pages,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=torch.float16,
        device=device,
    )
    v_cache = torch.randn_like(k_cache)
    out = torch.empty_like(q)
    lse = torch.empty((total_num_rows, num_qo_heads), dtype=torch.float32, device=device)
    float_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    int_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    pin_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)

    print_tensor("q", q)
    print_tensor("k_cache", k_cache)
    print_tensor("v_cache", v_cache)
    print_tensor("float_ws", float_ws, sample_count=0)
    print_tensor("int_ws", int_ws, sample_count=0)

    plan_args = {
        "total_num_rows": total_num_rows,
        "batch": batch,
        "num_qo_heads": num_qo_heads,
        "num_kv_heads": num_kv_heads,
        "page_size": page_size,
        "enable_cuda_graph": enable_cuda_graph,
        "head_dim_qk": head_dim,
        "head_dim_vo": head_dim,
        "causal": causal,
        "window_left": -1,
        "fixed_split_size": 0,
        "disable_split_kv": disable_split_kv,
        "enable_pdl": 0,
    }
    print(f"plan_args={plan_args}", flush=True)

    plan_info = mod.plan(
        float_ws,
        int_ws,
        pin_ws,
        qo_indptr_h,
        kv_indptr_h,
        kv_len_arr_h,
        total_num_rows,
        batch,
        num_qo_heads,
        num_kv_heads,
        page_size,
        enable_cuda_graph,
        head_dim,
        head_dim,
        causal,
        -1,
        0,
        disable_split_kv,
        0,
    )
    print(f"plan_info_type={type(plan_info)}", flush=True)
    print(f"plan_info={list(plan_info)}", flush=True)

    run_tail = xllm_tail_args(sm_scale)
    print(
        "paged_run_args:\n"
        f"    mask_mode={1 if causal else 0}\n"
        f"    layout={layout}\n"
        "    window_left=-1\n"
        "    enable_pdl=False\n"
        f"    tail={run_tail}",
        flush=True,
    )
    mod.paged_run(
        float_ws,
        int_ws,
        plan_info,
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        out,
        lse,
        1 if causal else 0,
        layout,
        -1,
        False,
        *run_tail,
    )
    torch.cuda.synchronize()

    ref = ref_paged_prefill(
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        causal,
        sm_scale,
        layout,
    )
    print_tensor("out", out)
    print_tensor("ref", ref)
    print_tensor("lse", lse)
    print_batch_diffs(out, ref, qo_indptr, num_qo_heads)

    diff = (out.float() - ref.float()).abs()
    print(
        f"global max_abs={float(diff.max().item()):.6g} "
        f"mean_abs={float(diff.mean().item()):.6g} "
        f"allclose={torch.allclose(out.float(), ref.float(), atol=4e-2, rtol=4e-2)}",
        flush=True,
    )
    assert torch.allclose(out.float(), ref.float(), atol=4e-2, rtol=4e-2)
