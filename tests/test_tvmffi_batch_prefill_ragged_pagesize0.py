import math
import os
import sys
from functools import lru_cache
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from flashinfer.jit import gen_jit_spec  # noqa: E402
from flashinfer.jit.attention.tvm import gen_customize_batch_prefill_tvm_binding  # noqa: E402
from test_tvmffi_batch_prefill import HND, NHD, make_indptr, ref_ragged_prefill  # noqa: E402


PLAN_PAGE_SIZE = int(os.environ.get("FLASHINFER_RAGGED_TEST_PLAN_PAGE_SIZE", "0"))
RAGGED_CASES = []


def _make_case(idx, head_dim, layout, causal, q_lens, kv_lens, num_qo_heads, num_kv_heads):
    return pytest.param(
        head_dim,
        layout,
        causal,
        q_lens,
        kv_lens,
        num_qo_heads,
        num_kv_heads,
        id=(
            f"{idx:03d}_h{head_dim}_{'nhd' if layout == NHD else 'hnd'}_"
            f"{'causal' if causal else 'noncausal'}_"
            f"b{len(q_lens)}_q{sum(q_lens)}_kv{sum(kv_lens)}_"
            f"heads{num_qo_heads}x{num_kv_heads}"
        ),
    )


_BASE_LENGTHS = [
    [1],
    [2, 3],
    [5, 1, 7],
    [17, 33],
    [65],
    [65, 129],
    [3, 5, 9, 1],
    [31, 64, 7],
    [66, 17],
    [127],
]
_HEAD_PAIRS = [(1, 1), (2, 2), (4, 4), (4, 2), (8, 2)]

for _idx in range(100):
    _head_dim = [64, 128][_idx % 2]
    _layout = [NHD, HND][(_idx // 2) % 2]
    _causal = (_idx % 3) != 1
    _q_lens = list(_BASE_LENGTHS[_idx % len(_BASE_LENGTHS)])
    if _causal:
        _kv_lens = list(_q_lens)
    else:
        _kv_lens = [q + 1 + ((_idx + pos) % 5) for pos, q in enumerate(_q_lens)]
    _num_qo_heads, _num_kv_heads = _HEAD_PAIRS[(_idx // len(_BASE_LENGTHS)) % len(_HEAD_PAIRS)]
    RAGGED_CASES.append(
        _make_case(
            _idx,
            _head_dim,
            _layout,
            _causal,
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
        uri=f"tvmffi_func_batch_prefill_fp16_h{head_dim}_ragged_pagesize0",
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


@pytest.mark.parametrize(
    "head_dim,layout,causal,q_lens,kv_lens,num_qo_heads,num_kv_heads",
    RAGGED_CASES,
)
def test_tvmffi_ragged_prefill_pagesize0(
    head_dim,
    layout,
    causal,
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
    )
    device = "cuda:0"
    mod = build_mod(head_dim)
    batch = len(q_lens)
    qo_indptr = make_indptr(q_lens, device)
    kv_indptr = make_indptr(kv_lens, device)
    qo_indptr_h = qo_indptr.cpu().contiguous()
    kv_indptr_h = kv_indptr.cpu().contiguous()
    kv_len_arr_h = torch.tensor(kv_lens, dtype=torch.int32).contiguous()
    total_num_rows = int(qo_indptr_h[-1].item())
    total_kv_rows = int(kv_indptr_h[-1].item())

    q = torch.randn(total_num_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device)
    if layout == NHD:
        k = torch.randn(total_kv_rows, num_kv_heads, head_dim, dtype=torch.float16, device=device)
    else:
        k = torch.randn(num_kv_heads, total_kv_rows, head_dim, dtype=torch.float16, device=device)
    v = torch.randn_like(k)
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
        PLAN_PAGE_SIZE,
        False,
        head_dim,
        head_dim,
        causal,
        -1,
        0,
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
        1 if causal else 0,
        layout,
        -1,
        False,
        *xllm_tail_args(sm_scale),
    )
    torch.cuda.synchronize()

    if layout == NHD:
        ref_k, ref_v = k, v
    else:
        ref_k, ref_v = k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
    ref = ref_ragged_prefill(q, ref_k, ref_v, qo_indptr, kv_indptr, causal, sm_scale)
    diff = (out.float() - ref.float()).abs()
    assert torch.allclose(out.float(), ref.float(), atol=4e-2, rtol=4e-2), (
        f"head_dim={head_dim} layout={layout} causal={causal} "
        f"q_lens={q_lens} kv_lens={kv_lens} heads={num_qo_heads}/{num_kv_heads} "
        f"plan_page_size={PLAN_PAGE_SIZE} shape={tuple(out.shape)} "
        f"max_abs={float(diff.max().item()):.6g} mean_abs={float(diff.mean().item()):.6g}"
    )
