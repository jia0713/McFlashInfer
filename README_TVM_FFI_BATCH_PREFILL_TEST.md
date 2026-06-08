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

## Installed Package Use

Framework integrations do not need a separate FlashInfer source checkout when
FlashInfer is installed as a package. The JIT path first looks for package data
under:

```text
flashinfer/data/tvm_binding
flashinfer/data/csrc
flashinfer/data/include
flashinfer/data/cutlass
```

When those package-data directories are absent, the same code falls back to the
source-tree layout used by this repo:

```text
tvm_binding
csrc
include
3rdparty/mcTlass
```

This keeps installed-package usage independent from the source checkout while
still allowing direct source-tree testing.

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

Run the checked-in test script:

```bash
python tests/test_tvmffi_batch_prefill.py
```

This script JIT-compiles the TVM-FFI batch prefill module, verifies the exported
`plan` and `paged_run` functions, and compares `paged_run` against a PyTorch
reference implementation.

Expected output has one line per case:

```text
nhd_causal_single_page: plan_info_len=15 out_shape=(3, 2, 64) ... allclose=True
nhd_noncausal_multi_page_gqa: plan_info_len=15 out_shape=(6, 4, 64) ... allclose=True
hnd_causal_multi_page_gqa: plan_info_len=15 out_shape=(5, 4, 64) ... allclose=True
nhd_causal_long_non64_gqa: plan_info_len=15 out_shape=(194, 4, 64) ... allclose=True
hnd_noncausal_long_non64_gqa: plan_info_len=15 out_shape=(203, 4, 64) ... allclose=True
ragged_causal_long_non64_gqa: plan_info_len=15 out_shape=(82, 4, 64) ... allclose=True
all cases passed
```

The exact error values may change slightly across toolchain versions.

## Notes

- The tested path is `plan + paged_run`.
- The tested layouts are paged KV `NHD` and `HND`.
- The tested dtype is `fp16`.
- The tested head dimension is `64`.
- The tested attention modes include causal and non-causal.
- The tested page table includes single-page, multi-page, and non-identity page
  indices.
- The tested head grouping includes MHA and GQA.
- The tested sequence lengths include long Q/KV lengths that are not divisible
  by 64, including KV lengths above 1K and 3K tokens.
- The checked-in test uses the same FA2 TVM-FFI additional argument tail that
  xLLM passes for `ragged_run` and `paged_run`: optional custom mask/alibi/prefix
  tensors plus logits/scale/RoPE scalar parameters.
- `ragged_run` is compiled, exported, and covered by a numeric correctness case.
- Inline RoPE is not covered by this minimal TVM-FFI test.
