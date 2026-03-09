/*
 * 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MLA_FA2_CUH_
#define FLASHINFER_MLA_FA2_CUH_

#include "mla_kernels_xcore1000.cuh"
#include "mla_kernels_xcore1500.cuh"

namespace flashinfer {

namespace mla {

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAPagedAttentionKernel(
    const Params params) {
#if (__MACA_ARCH__ == 1000)
  if constexpr (KTraits::CTA_TILE_Q == 64) {
    batch_mla_paged_attention_kernel_xc1000_ctq64<KTraits, Params>(params);
  } else if constexpr (KTraits::CTA_TILE_Q == 32) {
    batch_mla_paged_attention_kernel_xc1000_ctq32<KTraits, Params>(params);
  } else {
    FLASHINFER_RUNTIME_ASSERT("Unsupported CTA_TILE_Q");
  }
#elif (__MACA_ARCH__ == 1500)
  if constexpr (KTraits::NUM_STAGES == 1) {
    batch_mla_paged_attention_kernel_xc1500_1stage<KTraits, Params>(params);
  } else {
    batch_mla_paged_attention_kernel_xc1500_multistage<KTraits, Params>(params);
  }
#else
  FLASHINFER_RUNTIME_ASSERT("Unsupported MACA architecture");
#endif
}

#define DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, ...) \
  if (smem_limit_per_sm >= 128 * 1024) {                                                \
    constexpr uint32_t NUM_STAGES = 2;                                                  \
    constexpr uint32_t CTA_TILE_KV = 32;                                                \
    constexpr bool QK_SHARD = true;                                                     \
    __VA_ARGS__;                                                                        \
  } else if (smem_limit_per_sm >= 64 * 1024) {                                          \
    constexpr uint32_t NUM_STAGES = 1;                                                  \
    constexpr uint32_t CTA_TILE_KV = 32;                                                \
    constexpr bool QK_SHARD = true;                                                     \
    __VA_ARGS__;                                                                        \
  } else {                                                                              \
    std::ostringstream err;                                                             \
    err << "Unsupported shared memory size: " << smem_limit_per_sm;                     \
    FLASHINFER_ERROR(err.str());                                                        \
    return cudaErrorNotSupported;                                                       \
  }

template <MaskMode MASK_MODE, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, uint32_t CTA_TILE_Q,
          typename Params>
cudaError_t BatchMLAPagedAttention(Params params, uint32_t num_blks_x, uint32_t num_blks_y,
                                   cudaStream_t stream) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  if (MASK_MODE == MaskMode::kCustom) {
    return cudaErrorNotSupported;
  }
  constexpr bool CAUSAL = MASK_MODE == MaskMode::kCausal;

  dim3 nblks(num_blks_x, num_blks_y);
  dim3 nthrs(64, CTA_TILE_Q / 16, 2);
  // get GPU shared memory size
  int device;
  int smem_limit_per_sm;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&smem_limit_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);

  DISPATCH_SMEM_CONFIG(smem_limit_per_sm, NUM_STAGES, CTA_TILE_KV, QK_SHARD, {
    using KTraits = KernelTraits<CAUSAL, NUM_STAGES, QK_SHARD, HEAD_DIM_CKV, HEAD_DIM_KPE,
                                 CTA_TILE_Q, CTA_TILE_KV, DTypeQ, DTypeKV, DTypeO, IdType>;
    size_t smem_size = sizeof(typename KTraits::SharedStorage);
    auto kernel = BatchMLAPagedAttentionKernel<KTraits, Params>;
    void* args[] = {(void*)&params};

    FLASHINFER_CUDA_CALL(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    FLASHINFER_CUDA_CALL(
        cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
  });

  return cudaSuccess;
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_FA2_CUH_
