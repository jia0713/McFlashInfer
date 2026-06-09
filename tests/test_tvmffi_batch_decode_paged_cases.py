import math
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.jit import gen_jit_spec
from flashinfer.jit.attention.tvm import gen_customize_batch_decode_tvm_binding


NHD = 0
HND = 1
DECODE_CASES = []


def _make_case(
    idx,
    head_dim,
    layout,
    enable_cuda_graph,
    use_lse,
    page_size,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
):
    return pytest.param(
        head_dim,
        layout,
        enable_cuda_graph,
        use_lse,
        page_size,
        kv_lens,
        num_qo_heads,
        num_kv_heads,
        id=(
            f"{idx:03d}_h{head_dim}_{'nhd' if layout == NHD else 'hnd'}_"
            f"cg{int(enable_cuda_graph)}_lse{int(use_lse)}_ps{page_size}_"
            f"b{len(kv_lens)}_kv{sum(kv_lens)}_heads{num_qo_heads}x{num_kv_heads}"
        ),
    )


_BATCH_SIZES = [1, 2, 4, 8]
_PAGE_SIZES = [1, 4, 16, 64]
_LENGTH_VALUES = [1, 2, 7, 17, 33, 65, 129, 257]
_HEAD_PAIRS = [(1, 1), (2, 2), (4, 4), (4, 2), (8, 2)]


def _make_kv_lens(idx, batch_size):
    return [
        _LENGTH_VALUES[(idx + request_idx * 3 + batch_size) % len(_LENGTH_VALUES)]
        + (request_idx % 2)
        for request_idx in range(batch_size)
    ]


for _idx in range(96):
    _head_dim = [64, 128][_idx % 2]
    _layout = [NHD, HND][(_idx // 2) % 2]
    _enable_cuda_graph = ((_idx // 4) % 2) == 1
    _use_lse = ((_idx // 8) % 2) == 1
    _batch_size = _BATCH_SIZES[(_idx // 3) % len(_BATCH_SIZES)]
    _page_size = _PAGE_SIZES[(_idx // len(_BATCH_SIZES)) % len(_PAGE_SIZES)]
    _kv_lens = _make_kv_lens(_idx, _batch_size)
    _num_qo_heads, _num_kv_heads = _HEAD_PAIRS[
        (_idx // (len(_BATCH_SIZES) * len(_PAGE_SIZES))) % len(_HEAD_PAIRS)
    ]
    DECODE_CASES.append(
        _make_case(
            _idx,
            _head_dim,
            _layout,
            _enable_cuda_graph,
            _use_lse,
            _page_size,
            _kv_lens,
            _num_qo_heads,
            _num_kv_heads,
        )
    )


@lru_cache(maxsize=None)
def build_mod(head_dim):
    uri, sources = gen_customize_batch_decode_tvm_binding(
        uri=f"tvmffi_func_batch_decode_fp16_h{head_dim}_paged_cases",
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        dtype_o=torch.float16,
        idtype=torch.int32,
        head_dim_qk=head_dim,
        head_dim_vo=head_dim,
        additional_tensor_names=["maybe_alibi_slopes"],
        additional_tensor_dtypes=["float"],
        additional_scalar_names=[
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],
        additional_scalar_dtypes=["double", "double", "double", "double"],
        variant_name="DefaultAttention<false, false, false, false>",
        variant_decl="using namespace flashinfer;",
        pos_encoding_mode=0,
    )
    return gen_jit_spec(uri, sources).build_and_load_tvm_ffi()


def decode_tail_args(sm_scale):
    return [
        None,
        0.0,
        sm_scale,
        1.0,
        1.0 / 10000.0,
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


def ref_decode(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale, layout):
    out = torch.empty_like(q, dtype=torch.float32)
    batch = q.shape[0]
    for b in range(batch):
        p0, p1 = int(kv_indptr[b].item()), int(kv_indptr[b + 1].item())
        kb, vb = gather_kv(
            k_cache,
            v_cache,
            kv_indices,
            p0,
            p1,
            kv_last_page_len[b].item(),
            layout,
        )
        qb = q[b].float()
        kb = kb.float()
        vb = vb.float()
        group = qb.shape[0] // kb.shape[1]
        for h in range(qb.shape[0]):
            kh = h // group
            scores = qb[h] @ kb[:, kh].transpose(0, 1) * sm_scale
            probs = torch.softmax(scores, dim=-1)
            out[b, h] = probs @ vb[:, kh]
    return out.to(q.dtype)


@pytest.mark.parametrize(
    (
        "head_dim,layout,enable_cuda_graph,use_lse,page_size,"
        "kv_lens,num_qo_heads,num_kv_heads"
    ),
    DECODE_CASES,
)
def test_tvmffi_batch_decode_paged_cases(
    head_dim,
    layout,
    enable_cuda_graph,
    use_lse,
    page_size,
    kv_lens,
    num_qo_heads,
    num_kv_heads,
):
    torch.manual_seed(
        head_dim
        + layout * 13
        + len(kv_lens) * 19
        + num_qo_heads * 23
        + num_kv_heads * 29
        + int(enable_cuda_graph) * 31
        + int(use_lse) * 37
        + page_size * 41
        + sum(kv_lens) * 43
    )
    device = "cuda:0"
    mod = build_mod(head_dim)
    batch = len(kv_lens)
    kv_indptr, kv_indices, kv_last_page_len, total_pages = make_page_table(
        kv_lens, page_size, device
    )
    kv_indptr_h = kv_indptr.cpu().contiguous()

    q = torch.randn(batch, num_qo_heads, head_dim, dtype=torch.float16, device=device)
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
    lse = torch.empty((batch, num_qo_heads), dtype=torch.float32, device=device) if use_lse else None
    float_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    int_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    pin_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
    sm_scale = 1.0 / math.sqrt(head_dim)

    plan_info = mod.plan(
        float_ws,
        int_ws,
        pin_ws,
        kv_indptr_h,
        batch,
        num_qo_heads,
        num_kv_heads,
        page_size,
        enable_cuda_graph,
        -1,
        0.0,
        head_dim,
        head_dim,
        q,
        k_cache,
    )
    mod.run(
        float_ws,
        int_ws,
        plan_info,
        q,
        k_cache,
        v_cache,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        out,
        lse,
        layout,
        -1,
        False,
        *decode_tail_args(sm_scale),
    )
    torch.cuda.synchronize()

    ref = ref_decode(q, k_cache, v_cache, kv_indptr, kv_indices, kv_last_page_len, sm_scale, layout)
    diff = (out.float() - ref.float()).abs()
    assert torch.allclose(out.float(), ref.float(), atol=4e-2, rtol=4e-2), (
        f"head_dim={head_dim} layout={layout} enable_cuda_graph={enable_cuda_graph} "
        f"use_lse={use_lse} page_size={page_size} kv_lens={kv_lens} "
        f"heads={num_qo_heads}/{num_kv_heads} shape={tuple(out.shape)} "
        f"max_abs={float(diff.max().item()):.6g} mean_abs={float(diff.mean().item()):.6g}"
    )
