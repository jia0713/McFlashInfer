# TVM FFI Batch Prefill Test on MACA

This document records how to verify the TVM-FFI batch prefill path on the remote
MACA machine.

## Remote Environment

Remote host:

```bash
ssh acl_ici@10.0.180.24
```

Conda environment:

```bash
source ~/miniforge3/bin/activate fi
```

MACA environment:

```bash
export MACA_PATH=/opt/maca
export CUDA_PATH=${MACA_PATH}/tools/cu-bridge
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${MACA_PATH}/tools/cu-bridge/bin:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}
```

Use a local JIT cache under the repo, so the test does not depend on a shared
cache state:

```bash
export FLASHINFER_WORKSPACE_BASE=/home/acl_ici/workspace/McFlashInfer_tvmffi/.test-cache
```

## Prepare Source Tree

The tested source tree was placed at:

```bash
cd /home/acl_ici/workspace/McFlashInfer_tvmffi
```

If you need to refresh it from the local checkout:

```bash
rsync -a --delete \
  --exclude='.git' \
  --exclude='3rdparty/flashinfer/.git' \
  --exclude='flashinfer-jit-cache' \
  --exclude='flashinfer-cubin' \
  ./ acl_ici@10.0.180.24:/home/acl_ici/workspace/McFlashInfer_tvmffi/
```

## Verify tvm_ffi

Run:

```bash
python - <<'PY'
import tvm_ffi

print("tvm_ffi", tvm_ffi.__file__)
print("include", tvm_ffi.libinfo.find_include_path())
print("dlpack", tvm_ffi.libinfo.find_dlpack_include_path())
PY
```

The `tvm_ffi` package may print a warning about failing to build the optional
`torch-c-dlpack` allocator extension on the MACA PyTorch headers. This warning
does not block the FlashInfer TVM-FFI module compile, load, or batch prefill
test.

## Functional Test

Create the test script:

```bash
cat > test_tvmffi_batch_prefill.py <<'PY'
import math
import os

import torch
from flashinfer.jit import gen_jit_spec
from flashinfer.jit.attention.tvm import gen_customize_batch_prefill_tvm_binding


def build_mod():
    uri, sources = gen_customize_batch_prefill_tvm_binding(
        backend="fa2",
        uri="tvmffi_func_batch_prefill_fp16_h64",
        dtype_q=torch.float16,
        dtype_kv=torch.float16,
        dtype_o=torch.float16,
        idtype=torch.int32,
        head_dim_qk=64,
        head_dim_vo=64,
        additional_tensor_names=[],
        additional_tensor_dtypes=[],
        additional_scalar_names=["sm_scale"],
        additional_scalar_dtypes=["float"],
        variant_name="DefaultAttention<use_custom_mask, false, false, false>",
        variant_decl="using namespace flashinfer;",
    )
    return gen_jit_spec(uri, sources).build_and_load_tvm_ffi()


def gather_kv(k_cache, v_cache, indices, start_page, end_page, last_len, page_size):
    k_parts = []
    v_parts = []
    for ppos in range(start_page, end_page):
        page_idx = int(indices[ppos].item())
        valid = page_size if ppos + 1 < end_page else int(last_len)
        k_parts.append(k_cache[page_idx, :valid])
        v_parts.append(v_cache[page_idx, :valid])
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
):
    out = torch.empty_like(q, dtype=torch.float32)
    batch = qo_indptr.numel() - 1
    page_size = k_cache.shape[1]
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


def main():
    torch.manual_seed(0)
    device = "cuda:0"
    mod = build_mod()

    batch = 2
    num_qo_heads = 2
    num_kv_heads = 2
    head_dim = 64
    page_size = 4
    qo_indptr = torch.tensor([0, 1, 3], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1, 2], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_last_page_len = torch.tensor([1, 2], dtype=torch.int32, device=device)

    # PrefillPlan expects host indptr inputs in this McFlashInfer implementation.
    qo_indptr_h = qo_indptr.cpu().contiguous()
    kv_indptr_h = kv_indptr.cpu().contiguous()
    kv_len_arr_h = (
        page_size * (kv_indptr_h[1:] - kv_indptr_h[:-1] - 1)
        + kv_last_page_len.cpu()
    ).contiguous()
    total_num_rows = int(qo_indptr_h[-1].item())

    q = torch.randn(
        total_num_rows, num_qo_heads, head_dim, dtype=torch.float16, device=device
    )
    k_cache = torch.randn(
        2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    v_cache = torch.randn(
        2, page_size, num_kv_heads, head_dim, dtype=torch.float16, device=device
    )
    out = torch.empty_like(q)
    lse = torch.empty((total_num_rows, num_qo_heads), dtype=torch.float32, device=device)

    float_ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    int_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    pin_ws = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)

    causal = True
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
        False,
        head_dim,
        head_dim,
        causal,
        -1,
        0,
        False,
        0,
    )
    print("plan_info_len", len(plan_info), flush=True)

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
        1,
        0,
        -1,
        False,
        sm_scale,
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
    )
    diff = (out.float() - ref.float()).abs()
    print("out_shape", tuple(out.shape), flush=True)
    print("max_abs", float(diff.max().item()), flush=True)
    print("mean_abs", float(diff.mean().item()), flush=True)
    print(
        "allclose",
        bool(torch.allclose(out.float(), ref.float(), atol=3e-2, rtol=3e-2)),
        flush=True,
    )
    print("out_sample", out.flatten()[:8].detach().cpu().tolist(), flush=True)
    print("ref_sample", ref.flatten()[:8].detach().cpu().tolist(), flush=True)

    # Avoid tvm_ffi/Python shutdown-time destructor issues observed in this env.
    os._exit(0)


if __name__ == "__main__":
    main()
PY
```

Run:

```bash
python test_tvmffi_batch_prefill.py
```

Expected output:

```text
plan_info_len 15
out_shape (3, 2, 64)
max_abs 0.0009765625
mean_abs 8.702278137207031e-06
allclose True
```

The exact sample values may change if the seed or test shape is changed.

## Notes

- The tested path is `plan + paged_run`.
- The tested layout is paged KV `NHD`.
- The tested dtype is `fp16`.
- The tested head dimension is `64`.
- `ragged_run` is compiled and exported, but this test does not validate ragged
  numeric correctness.
- Inline RoPE is not covered by this minimal TVM-FFI test.
