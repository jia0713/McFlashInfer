import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.jit import gen_jit_spec
from flashinfer.jit.attention.tvm import gen_customize_batch_prefill_tvm_binding


NHD = 0
HND = 1


@dataclass
class Case:
    name: str
    layout: int
    causal: bool
    q_lens: list[int]
    kv_lens: list[int]
    num_qo_heads: int
    num_kv_heads: int
    page_size: int


def build_mod():
    uri, sources = gen_customize_batch_prefill_tvm_binding(
        backend="fa2",
        uri="tvmffi_func_batch_prefill_fp16_h64_multi_case",
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        dtype_o=torch.float16,
        idtype=torch.int32,
        head_dim_qk=64,
        head_dim_vo=64,
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
    # Reverse the physical page order to cover non-identity page indices.
    kv_indices = torch.flip(page_ids, dims=[0]).contiguous()
    last_page_len = [
        length % page_size if length % page_size != 0 else page_size for length in kv_lens
    ]
    kv_last_page_len = torch.tensor(last_page_len, dtype=torch.int32, device=device)
    return kv_indptr, kv_indices, kv_last_page_len, total_pages


def gather_kv(k_cache, v_cache, indices, start_page, end_page, last_len, page_size, layout):
    k_parts = []
    v_parts = []
    for ppos in range(start_page, end_page):
        page_idx = int(indices[ppos].item())
        valid = page_size if ppos + 1 < end_page else int(last_len)
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
    page_size = k_cache.shape[1] if layout == NHD else k_cache.shape[2]
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
            page_size,
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


def ref_ragged_prefill(q, k, v, qo_indptr, kv_indptr, causal, sm_scale):
    out = torch.empty_like(q, dtype=torch.float32)
    batch = qo_indptr.numel() - 1
    for b in range(batch):
        q0, q1 = int(qo_indptr[b].item()), int(qo_indptr[b + 1].item())
        k0, k1 = int(kv_indptr[b].item()), int(kv_indptr[b + 1].item())
        qb = q[q0:q1].float()
        kb = k[k0:k1].float()
        vb = v[k0:k1].float()
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


def run_case(mod, case, device):
    head_dim = 64
    batch = len(case.q_lens)
    qo_indptr = make_indptr(case.q_lens, device)
    kv_indptr, kv_indices, kv_last_page_len, total_pages = make_page_table(
        case.kv_lens, case.page_size, device
    )

    qo_indptr_h = qo_indptr.cpu().contiguous()
    kv_indptr_h = kv_indptr.cpu().contiguous()
    kv_len_arr_h = torch.tensor(case.kv_lens, dtype=torch.int32).contiguous()
    total_num_rows = int(qo_indptr_h[-1].item())

    q = torch.randn(
        total_num_rows, case.num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    if case.layout == NHD:
        k_cache = torch.randn(
            total_pages,
            case.page_size,
            case.num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
        v_cache = torch.randn_like(k_cache)
    else:
        k_cache = torch.randn(
            total_pages,
            case.num_kv_heads,
            case.page_size,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
        v_cache = torch.randn_like(k_cache)

    out = torch.empty_like(q)
    lse = torch.empty((total_num_rows, case.num_qo_heads), dtype=torch.float32, device=device)
    float_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    int_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    pin_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # PrefillPlan expects host indptr inputs in this McFlashInfer implementation.
    plan_info = mod.plan(
        float_ws,
        int_ws,
        pin_ws,
        qo_indptr_h,
        kv_indptr_h,
        kv_len_arr_h,
        total_num_rows,
        batch,
        case.num_qo_heads,
        case.num_kv_heads,
        case.page_size,
        False,
        head_dim,
        head_dim,
        case.causal,
        -1,
        0,
        False,
        0,
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
        1 if case.causal else 0,
        case.layout,
        -1,
        False,
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
        case.causal,
        sm_scale,
        case.layout,
    )
    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ok = torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    print(
        f"{case.name}: plan_info_len={len(plan_info)} "
        f"out_shape={tuple(out.shape)} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} allclose={bool(ok)}",
        flush=True,
    )
    if not ok:
        print("out_sample", out.flatten()[:8].detach().cpu().tolist(), flush=True)
        print("ref_sample", ref.flatten()[:8].detach().cpu().tolist(), flush=True)
        raise AssertionError(f"{case.name} failed")


def run_ragged_case(mod, device):
    head_dim = 64
    q_lens = [65, 17]
    kv_lens = [1025, 33]
    batch = len(q_lens)
    num_qo_heads = 4
    num_kv_heads = 2
    qo_indptr = make_indptr(q_lens, device)
    kv_indptr = make_indptr(kv_lens, device)
    qo_indptr_h = qo_indptr.cpu().contiguous()
    kv_indptr_h = kv_indptr.cpu().contiguous()
    kv_len_arr_h = torch.tensor(kv_lens, dtype=torch.int32).contiguous()
    total_num_rows = int(qo_indptr_h[-1].item())
    total_kv_rows = int(kv_indptr_h[-1].item())

    q = torch.randn(total_num_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    k = torch.randn(total_kv_rows, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    lse = torch.empty((total_num_rows, num_qo_heads), dtype=torch.float32, device=device)
    float_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    int_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    pin_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    sm_scale = 1.0 / math.sqrt(head_dim)
    causal = True

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
        1,
        False,
        head_dim,
        head_dim,
        causal,
        -1,
        -1,
        False,
        0,
    )

    mod.ragged_run(
        float_ws,
        int_ws,
        plan_info,
        q,
        k,
        v,
        qo_indptr,
        kv_indptr,
        out,
        lse,
        1,
        NHD,
        -1,
        False,
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
    )
    torch.cuda.synchronize()

    ref = ref_ragged_prefill(q, k, v, qo_indptr, kv_indptr, causal, sm_scale)
    diff = (out.float() - ref.float()).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    ok = torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    print(
        f"ragged_causal_long_non64_gqa: plan_info_len={len(plan_info)} "
        f"out_shape={tuple(out.shape)} max_abs={max_abs:.6g} "
        f"mean_abs={mean_abs:.6g} allclose={bool(ok)}",
        flush=True,
    )
    if not ok:
        raise AssertionError("ragged_causal_long_non64_gqa failed")


def main():
    torch.manual_seed(0)
    device = "cuda:0"
    mod = build_mod()
    cases = [
        Case(
            name="nhd_causal_single_page",
            layout=NHD,
            causal=True,
            q_lens=[1, 2],
            kv_lens=[1, 2],
            num_qo_heads=2,
            num_kv_heads=2,
            page_size=4,
        ),
        Case(
            name="nhd_noncausal_multi_page_gqa",
            layout=NHD,
            causal=False,
            q_lens=[2, 1, 3],
            kv_lens=[5, 3, 7],
            num_qo_heads=4,
            num_kv_heads=2,
            page_size=4,
        ),
        Case(
            name="hnd_causal_multi_page_gqa",
            layout=HND,
            causal=True,
            q_lens=[3, 2],
            kv_lens=[6, 4],
            num_qo_heads=4,
            num_kv_heads=2,
            page_size=3,
        ),
        Case(
            name="nhd_causal_long_non64_gqa",
            layout=NHD,
            causal=True,
            q_lens=[65, 129],
            kv_lens=[1025, 2049],
            num_qo_heads=4,
            num_kv_heads=2,
            page_size=16,
        ),
        Case(
            name="hnd_noncausal_long_non64_gqa",
            layout=HND,
            causal=False,
            q_lens=[67, 131, 5],
            kv_lens=[1023, 3073, 257],
            num_qo_heads=4,
            num_kv_heads=2,
            page_size=32,
        ),
    ]
    for case in cases:
        run_case(mod, case, device)
    run_ragged_case(mod, device)
    print("all cases passed", flush=True)
    # Avoid tvm_ffi/Python shutdown-time destructor issues observed in this env.
    os._exit(0)


if __name__ == "__main__":
    main()
