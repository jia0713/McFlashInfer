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
PAGED_CASES = []


def _make_case(
    idx,
    head_dim,
    layout,
    causal,
    enable_cuda_graph,
    disable_split_kv,
    page_size,
    q_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
):
    return pytest.param(
        head_dim,
        layout,
        causal,
        enable_cuda_graph,
        disable_split_kv,
        page_size,
        q_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads,
        id=(
            f"{idx:03d}_h{head_dim}_{'nhd' if layout == NHD else 'hnd'}_"
            f"{'causal' if causal else 'noncausal'}_"
            f"cg{int(enable_cuda_graph)}_ds{int(disable_split_kv)}_"
            f"ps{page_size}_b{len(q_lens)}_q{sum(q_lens)}_kv{sum(kv_lens)}_"
            f"heads{num_qo_heads}x{num_kv_heads}"
        ),
    )


_BATCH_SIZES = [1, 2, 4]
_PAGE_SIZES = [1, 4, 16, 64]
_LENGTH_VALUES = [1, 2, 8, 17, 33, 65, 129, 257]
_HEAD_PAIRS = [(1, 1), (2, 2), (4, 4), (4, 2), (8, 2)]
_BOOL_CASES = [
    (True, False, False),
    (False, False, False),
    (True, True, False),
    (False, True, False),
    (True, False, True),
    (False, False, True),
]


def _make_q_lens(idx, batch_size, causal):
    if not causal:
        return [1 for _ in range(batch_size)]
    return [
        _LENGTH_VALUES[(idx + request_idx * 3 + batch_size) % len(_LENGTH_VALUES)]
        for request_idx in range(batch_size)
    ]


def _make_kv_lens(idx, q_lens, causal):
    if causal:
        return [q + ((_idx_mod + 1) % 4) for _idx_mod, q in enumerate(q_lens)]
    return [
        _LENGTH_VALUES[(idx + request_idx * 5 + 2) % len(_LENGTH_VALUES)]
        + request_idx
        for request_idx in range(len(q_lens))
    ]


for _idx in range(120):
    _head_dim = [64, 128][_idx % 2]
    _layout = [NHD, HND][(_idx // 2) % 2]
    _causal, _enable_cuda_graph, _disable_split_kv = _BOOL_CASES[
        (_idx // 4) % len(_BOOL_CASES)
    ]
    _batch_size = _BATCH_SIZES[_idx % len(_BATCH_SIZES)]
    _page_size = _PAGE_SIZES[(_idx // len(_BATCH_SIZES)) % len(_PAGE_SIZES)]
    _q_lens = _make_q_lens(_idx, _batch_size, _causal)
    _kv_lens = _make_kv_lens(_idx, _q_lens, _causal)
    _num_qo_heads, _num_kv_heads = _HEAD_PAIRS[
        (_idx // (len(_BATCH_SIZES) * len(_PAGE_SIZES))) % len(_HEAD_PAIRS)
    ]
    PAGED_CASES.append(
        _make_case(
            _idx,
            _head_dim,
            _layout,
            _causal,
            _enable_cuda_graph,
            _disable_split_kv,
            _page_size,
            _q_lens,
            _kv_lens,
            _num_qo_heads,
            _num_kv_heads,
        )
    )


@lru_cache(maxsize=None)
def build_mod(head_dim):
    uri, sources = gen_customize_batch_prefill_tvm_binding(
        backend="fa2",
        uri=f"tvmffi_func_batch_prefill_fp16_h{head_dim}_paged_cases",
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


@pytest.mark.parametrize(
    (
        "head_dim,layout,causal,enable_cuda_graph,disable_split_kv,"
        "page_size,q_lens,kv_lens,num_qo_heads,num_kv_heads"
    ),
    PAGED_CASES,
)
def test_tvmffi_paged_prefill_cases(
    head_dim,
    layout,
    causal,
    enable_cuda_graph,
    disable_split_kv,
    page_size,
    q_lens,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
):
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
    device = "cuda:0"
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

    q = torch.randn(total_num_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    if layout == NHD:
        k_cache = torch.randn(
            total_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.float16,
            device=device,
        )
    else:
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
    sm_scale = 1.0 / math.sqrt(head_dim)

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
        *xllm_tail_args(sm_scale),
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
    diff = (out.float() - ref.float()).abs()
    assert torch.allclose(out.float(), ref.float(), atol=4e-2, rtol=4e-2), (
        f"head_dim={head_dim} layout={layout} causal={causal} "
        f"enable_cuda_graph={enable_cuda_graph} disable_split_kv={disable_split_kv} "
        f"page_size={page_size} q_lens={q_lens} kv_lens={kv_lens} "
        f"heads={num_qo_heads}/{num_kv_heads} shape={tuple(out.shape)} "
        f"max_abs={float(diff.max().item()):.6g} mean_abs={float(diff.mean().item()):.6g}"
    )
