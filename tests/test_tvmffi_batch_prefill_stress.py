import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

_TESTS_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_ROOT.parents[0]
sys.path.insert(0, str(_TESTS_ROOT))
sys.path.insert(1, str(_REPO_ROOT))

from test_tvmffi_batch_prefill import (  # noqa: E402
    HND,
    NHD,
    build_mod,
    make_indptr,
    make_page_table,
    ref_paged_prefill,
    ref_ragged_prefill,
)


@dataclass
class StressCase:
    name: str
    op: str
    layout: int
    causal: bool
    q_lens: list[int]
    kv_lens: list[int]
    num_qo_heads: int
    num_kv_heads: int
    page_size: int


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


def run_paged_case(mod, case, device):
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

    plan_info = mod.plan(
        float_ws,
        int_ws,
        pin_ws,
        qo_indptr_h,
        kv_indptr_h,
        kv_len_arr_h,
        total_num_rows if case.causal else batch,
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
        case.causal,
        sm_scale,
        case.layout,
    )
    diff = (out.float() - ref.float()).abs()
    ok = torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    return ok, float(diff.max().item()), float(diff.mean().item()), tuple(out.shape)


def run_ragged_case(mod, case, device):
    head_dim = 64
    batch = len(case.q_lens)
    qo_indptr = make_indptr(case.q_lens, device)
    kv_indptr = make_indptr(case.kv_lens, device)
    qo_indptr_h = qo_indptr.cpu().contiguous()
    kv_indptr_h = kv_indptr.cpu().contiguous()
    kv_len_arr_h = torch.tensor(case.kv_lens, dtype=torch.int32).contiguous()
    total_num_rows = int(qo_indptr_h[-1].item())
    total_kv_rows = int(kv_indptr_h[-1].item())

    q = torch.randn(
        total_num_rows, case.num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    if case.layout == NHD:
        k = torch.randn(
            total_kv_rows, case.num_kv_heads, head_dim, dtype=torch.float16, device=device
        )
    else:
        k = torch.randn(
            case.num_kv_heads, total_kv_rows, head_dim, dtype=torch.float16, device=device
        )
    v = torch.randn_like(k)
    out = torch.empty_like(q)
    lse = torch.empty((total_num_rows, case.num_qo_heads), dtype=torch.float32, device=device)
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
        case.num_qo_heads,
        case.num_kv_heads,
        1,
        False,
        head_dim,
        head_dim,
        case.causal,
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
        1 if case.causal else 0,
        case.layout,
        -1,
        False,
        *xllm_tail_args(sm_scale),
    )
    torch.cuda.synchronize()

    if case.layout == NHD:
        ref_k, ref_v = k, v
    else:
        ref_k, ref_v = k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()
    ref = ref_ragged_prefill(
        q, ref_k, ref_v, qo_indptr, kv_indptr, case.causal, sm_scale
    )
    diff = (out.float() - ref.float()).abs()
    ok = torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2)
    return ok, float(diff.max().item()), float(diff.mean().item()), tuple(out.shape)


def make_cases():
    cases = []
    batches = [[1], [2, 1], [3, 5, 2], [7, 1, 9, 4]]
    page_sizes = [1, 2, 3, 4, 7, 8, 16, 32]
    layouts = [NHD, HND]
    head_pairs = [(2, 2), (4, 2)]

    for i in range(120):
        q_lens = batches[i % len(batches)]
        page_size = page_sizes[i % len(page_sizes)]
        causal = (i % 2) == 0
        if causal:
            kv_lens = [q + (i % 5) * page_size for q in q_lens]
        else:
            # xLLM uses non-causal paged prefill as the tensor-core decode path,
            # where each request contributes one query token.
            q_lens = [1 for _ in q_lens]
            kv_lens = [q + 1 + (i % 7) for q in q_lens]
        num_qo_heads, num_kv_heads = head_pairs[i % len(head_pairs)]
        cases.append(
            StressCase(
                name=f"paged_{i:03d}",
                op="paged",
                layout=layouts[i % len(layouts)],
                causal=causal,
                q_lens=q_lens,
                kv_lens=kv_lens,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                page_size=page_size,
            )
        )

    long_paged = [
        ([65, 129], [1025, 2049], 16, True, NHD),
        ([1, 1, 1], [1023, 3073, 257], 32, False, HND),
        ([1, 1, 1, 1], [257, 513, 1025, 33], 16, False, NHD),
        ([33, 65], [97, 257], 7, True, HND),
    ]
    for i, (q_lens, kv_lens, page_size, causal, layout) in enumerate(long_paged):
        cases.append(
            StressCase(
                name=f"paged_long_{i:03d}",
                op="paged",
                layout=layout,
                causal=causal,
                q_lens=q_lens,
                kv_lens=kv_lens,
                num_qo_heads=4,
                num_kv_heads=2,
                page_size=page_size,
            )
        )

    ragged_lengths = [[1], [2, 3], [5, 1, 7], [17, 33], [65], [65, 129]]
    for i in range(76):
        q_lens = ragged_lengths[i % len(ragged_lengths)]
        causal = (i % 2) == 0
        # Ragged causal models regular prefill, so q_len == kv_len. q_len < kv_len
        # belongs to paged/chunked prefill in xLLM.
        kv_lens = q_lens if causal else [q + 1 + (i % 4) for q in q_lens]
        num_qo_heads, num_kv_heads = head_pairs[i % len(head_pairs)]
        cases.append(
            StressCase(
                name=f"ragged_{i:03d}",
                op="ragged",
                layout=layouts[i % len(layouts)],
                causal=causal,
                q_lens=q_lens,
                kv_lens=kv_lens,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                page_size=1,
            )
        )
    return cases


def main():
    torch.manual_seed(0)
    device = "cuda:0"
    mod = build_mod()
    cases = make_cases()
    failures = []
    for idx, case in enumerate(cases, 1):
        if case.op == "paged":
            ok, max_abs, mean_abs, shape = run_paged_case(mod, case, device)
        else:
            ok, max_abs, mean_abs, shape = run_ragged_case(mod, case, device)
        if (idx % 20) == 0 or not ok:
            print(
                f"{idx:03d}/{len(cases)} {case.name} {case.op} "
                f"shape={shape} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} "
                f"allclose={bool(ok)}",
                flush=True,
            )
        if not ok:
            failures.append((case, max_abs, mean_abs, shape))
    print(f"stress_total {len(cases)}", flush=True)
    print(f"stress_failures {len(failures)}", flush=True)
    if failures:
        for case, max_abs, mean_abs, shape in failures[:10]:
            print(
                f"FAILED {case.name} op={case.op} layout={case.layout} "
                f"causal={case.causal} q_lens={case.q_lens} kv_lens={case.kv_lens} "
                f"heads={case.num_qo_heads}/{case.num_kv_heads} page_size={case.page_size} "
                f"shape={shape} max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}",
                flush=True,
            )
        raise AssertionError(f"{len(failures)} stress cases failed")
    print("stress all cases passed", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
