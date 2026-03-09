/*
 * Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 */
#ifndef FLASHINFER_PREFILL_UTILS_CUH_
#define FLASHINFER_PREFILL_UTILS_CUH_

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include "../cp_async.cuh"
#include "../fastdiv.cuh"
#include "../frag_layout_swizzle.cuh"
#include "../math.cuh"
#include "../mma.cuh"
#include "../page.cuh"
#include "../permuted_smem.cuh"
#include "../pos_enc.cuh"
#include "../utils.cuh"
#include "cascade.cuh"
#include "mask.cuh"
#include "variants.cuh"
namespace flashinfer {

DEFINE_HAS_MEMBER(maybe_q_rope_offset)
DEFINE_HAS_MEMBER(maybe_k_rope_offset)

namespace cg = cooperative_groups;
using cp_async::SharedMemFillMode;
using mma::MMAMode;

constexpr uint32_t WARP_SIZE = 64;

constexpr uint32_t get_num_warps_q(const uint32_t cta_tile_q) {
  if (cta_tile_q > 64) {
    return 8;
  } else if (cta_tile_q > 32) {
    return 4;
  } else if (cta_tile_q > 16) {
    return 2;
  } else {
    return 1;
  }
}

constexpr uint32_t get_num_warps_kv(const uint32_t cta_tile_kv) { return 1; }

constexpr uint32_t get_num_mma_q(const uint32_t cta_tile_q) {
  return cta_tile_q / 16 / get_num_warps_q(cta_tile_q);
}

template <uint32_t NUM_WARPS_KV, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_QK,
          uint32_t HEAD_DIM_VO, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];
    };
    struct {  // NOTE(Zihao): synchronize attention states across warps
      alignas(
          16) std::conditional_t<NUM_WARPS_KV == 1, float[1],
                                 float[NUM_WARPS_KV * CTA_TILE_Q * HEAD_DIM_VO]> cta_sync_o_smem;
      alignas(16) std::conditional_t<NUM_WARPS_KV == 1, float2[1],
                                     float2[NUM_WARPS_KV * CTA_TILE_Q]> cta_sync_md_smem;
    };
    alignas(16) DTypeQ q_smem[CTA_TILE_Q * HEAD_DIM_QK];
    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];
  };
};

template <MaskMode MASK_MODE_, uint32_t CTA_TILE_Q_, uint32_t NUM_MMA_Q_, uint32_t NUM_MMA_KV_,
          uint32_t NUM_MMA_D_QK_, uint32_t NUM_MMA_D_VO_, uint32_t NUM_WARPS_Q_,
          uint32_t NUM_WARPS_KV_, PosEncodingMode POS_ENCODING_MODE_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename DTypeQKAccum_, typename IdType_,
          typename AttentionVariant_>
struct KernelTraits {
  static constexpr MaskMode MASK_MODE = MASK_MODE_;
  static constexpr uint32_t NUM_MMA_Q = NUM_MMA_Q_;
  static constexpr uint32_t NUM_MMA_KV = NUM_MMA_KV_;
  static constexpr uint32_t NUM_MMA_D_QK = NUM_MMA_D_QK_;
  static constexpr uint32_t NUM_MMA_D_VO = NUM_MMA_D_VO_;
  static constexpr uint32_t NUM_WARPS_Q = NUM_WARPS_Q_;
  static constexpr uint32_t NUM_WARPS_KV = NUM_WARPS_KV_;
  static constexpr uint32_t NUM_THREADS = NUM_WARPS_Q * NUM_WARPS_KV * WARP_SIZE;
  static constexpr uint32_t NUM_WARPS = NUM_WARPS_Q * NUM_WARPS_KV;
  static constexpr uint32_t HEAD_DIM_QK = NUM_MMA_D_QK * 16;
  static constexpr uint32_t HEAD_DIM_VO = NUM_MMA_D_VO * 16;
  static constexpr uint32_t UPCAST_STRIDE_Q = HEAD_DIM_QK / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_Q_64B = HEAD_DIM_QK / upcast_size_64b<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_K = HEAD_DIM_QK / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_K_64B = HEAD_DIM_QK / upcast_size_64b<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_V = HEAD_DIM_VO / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_V_64B = HEAD_DIM_VO / upcast_size_64b<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_O = HEAD_DIM_VO / upcast_size<DTypeO_>();
  static constexpr uint32_t UPCAST_STRIDE_O_64B = HEAD_DIM_VO / upcast_size_64b<DTypeO_>();
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = NUM_MMA_KV * NUM_WARPS_KV * 16;

  static constexpr SwizzleMode SWIZZLE_MODE_Q = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KV =
      (sizeof(DTypeKV_) == 1 && HEAD_DIM_VO == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  static constexpr uint32_t K_THR_LAYOUT_ROW = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 8 : 16;
  static constexpr uint32_t K_THR_LAYOUT_COL = SWIZZLE_MODE_KV == SwizzleMode::k128B ? 8 : 4;
#if defined(__MACA_ARCH__) && (__MACA_ARCH__ == 1500 || __MACA_ARCH__ == 1600)
  static constexpr uint32_t V_THR_LAYOUT_ROW = 8;
  static constexpr uint32_t V_THR_LAYOUT_COL = 8;
#else
  // ldg-f16-4x4 pattern for v
  static constexpr uint32_t V_THR_LAYOUT_ROW =
      SWIZZLE_MODE_KV == SwizzleMode::k128B ? (CTA_TILE_KV == 32 ? 8 : 4) : 8;
  static constexpr uint32_t V_THR_LAYOUT_COL =
      SWIZZLE_MODE_KV == SwizzleMode::k128B ? (CTA_TILE_KV == 32 ? 8 : 16) : 8;
#endif

  static constexpr PosEncodingMode POS_ENCODING_MODE = POS_ENCODING_MODE_;
  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using DTypeQKAccum = DTypeQKAccum_;
  using IdType = IdType_;
  using AttentionVariant = AttentionVariant_;

  static constexpr bool IsInvalid() {
    return ((NUM_MMA_D_VO < 4) || (NUM_MMA_D_VO == 4 && NUM_MMA_KV % 2 == 1) ||
            (POS_ENCODING_MODE == PosEncodingMode::kRoPELlama && NUM_MMA_D_VO > 4 &&
             NUM_MMA_D_VO % (2 * NUM_WARPS_Q) != 0) ||
            (NUM_MMA_Q * (8 * NUM_MMA_D_VO + 2 * sizeof(DTypeQKAccum) * NUM_MMA_KV) >= 256) ||
            (sizeof(DTypeKV) == 1 && NUM_MMA_KV * 2 % NUM_WARPS_Q != 0) ||
            (sizeof(DTypeKV) == 1 && POS_ENCODING_MODE == PosEncodingMode::kRoPELlama));
  }

  using SharedStorage = SharedStorageQKVO<NUM_WARPS_KV, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_QK,
                                          HEAD_DIM_VO, DTypeQ, DTypeKV, DTypeO>;

  static constexpr DTypeQKAccum MaskFillValue =
      AttentionVariant::use_softmax ? DTypeQKAccum(-math::inf) : DTypeQKAccum(0.f);
};

namespace {

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_q() {
  if constexpr (KTraits::NUM_WARPS_Q == 1) {
    return 0;
  } else {
    return threadIdx.y;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_kv() {
  if constexpr (KTraits::NUM_WARPS_KV == 1) {
    return 0;
  } else {
    return threadIdx.z;
  }
}

template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx() {
  return get_warp_idx_kv<KTraits>() * KTraits::NUM_WARPS_Q + get_warp_idx_q<KTraits>();
}

/*!
 * \brief Apply Llama style rotary embedding to two 16x16 fragments.
 * \tparam T The data type of the input fragments.
 * \param x_first_half First fragment x[offset:offset+16, j*16:(j+1)*16]
 * \param x_second_half Second fragment x[offset:offset*16, j*16+d/2:(j+1)*16+d/2]
 * \param rope_freq Rope frequency
 * \param offset The offset of the first row in both fragments.
 * \note The sin/cos computation is slow, especially for A100 GPUs which has low
 *   non tensor-ops flops, will optimize in the future.
 */
template <typename T>
__device__ __forceinline__ void k_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t kv_offset) {
  static_assert(sizeof(T) == 2);
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 2 3
    // ---------
    // 4 5 | 6 7
    uint32_t i = reg_id / 4, j = (reg_id % 4) / 2;
    __sincosf(float(kv_offset + 8 * i) * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T>
__device__ __forceinline__ void q_frag_apply_llama_rope(T* x_first_half, T* x_second_half,
                                                        const float* rope_freq,
                                                        const uint32_t qo_packed_offset,
                                                        const uint_fastdiv group_size) {
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(float((qo_packed_offset + 8 * i) / group_size) * rope_freq[2 * j + reg_id % 2], &sin,
              &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

template <typename T, typename IdType>
__device__ __forceinline__ void q_frag_apply_llama_rope_with_pos(T* x_first_half, T* x_second_half,
                                                                 const float* rope_freq,
                                                                 const uint32_t qo_packed_offset,
                                                                 const uint_fastdiv group_size,
                                                                 const IdType* q_rope_offset) {
  float pos[2] = {static_cast<float>(q_rope_offset[qo_packed_offset / group_size]),
                  static_cast<float>(q_rope_offset[(qo_packed_offset + 8) / group_size])};
#pragma unroll
  for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
    float cos, sin, tmp;
    // 0 1 | 4 5
    // ---------
    // 2 3 | 6 7
    uint32_t i = ((reg_id % 4) / 2), j = (reg_id / 4);
    __sincosf(pos[i] * rope_freq[2 * j + reg_id % 2], &sin, &cos);
    tmp = x_first_half[reg_id];
    x_first_half[reg_id] = (tmp * cos - (float)x_second_half[reg_id] * sin);
    x_second_half[reg_id] = ((float)x_second_half[reg_id] * cos + tmp * sin);
  }
}

/*!
 * \brief Produce k/v fragments from global memory to shared memory.
 * \tparam NUM_MMA_D_VO The number of fragments in y dimension.
 * \tparam NUM_MMA_KV The number of fragments in z dimension.
 * \tparam num_warps The number of warps in the threadblock.
 * \tparam T The data type of the input tensor.
 * \param smem The shared memory to store kv fragments.
 * \param gptr The global memory pointer.
 * \param kv_idx_base The base kv index.
 * \param kv_len The length of kv tensor.
 */
template <bool produce_v, typename KTraits>
__device__ __forceinline__ void produce_kv(smem_t<KTraits::SWIZZLE_MODE_KV> smem,
                                           uint32_t* smem_offset, typename KTraits::DTypeKV** gptr,
                                           const uint32_t stride_n, const uint32_t kv_idx_base,
                                           const uint32_t kv_len) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 8;
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
        smem.template load_128b_async<DTypeKV, false>(*smem_offset, *gptr, kv_idx < kv_len);
        *smem_offset += 64;
        *gptr += 8 * upcast_size<DTypeKV>();
      }
      kv_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset) -
          16 * NUM_MMA_D;
      *gptr = *gptr + NUM_WARPS * 8 * stride_n - 2 * NUM_MMA_D * upcast_size<DTypeKV>();
    }
    *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_k_r(
    typename KTraits::DTypeKV** gptr, const uint32_t stride_n, const uint32_t k_idx_base,
    const uint32_t kv_len,
    uint32_t (*frag)[KTraits::NUM_MMA_D_QK / (8 / sizeof(typename KTraits::DTypeKV))][4]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    // using swizzle pattern <3, 3, 3>
    uint32_t k_idx = k_idx_base + warp_idx * 8 + lane_idx / 8;  // row idx
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
        cp_async::load_128b_pred(frag[i][j], *gptr, k_idx < kv_len);
        *gptr += 8 * upcast_size<DTypeKV>();
      }
      k_idx += NUM_WARPS * 8;
      *gptr += NUM_WARPS * 8 * stride_n - sizeof(DTypeKV) * NUM_MMA_D * upcast_size<DTypeKV>();
    }
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_k_r_64b(typename KTraits::DTypeKV** gptr,
                                                const uint32_t stride_n, const uint32_t k_idx_base,
                                                const uint32_t kv_len,
                                                uint32_t (*frag)[KTraits::NUM_MMA_D_QK / 4][2]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;

  static_assert(NUM_MMA_KV * 2 % NUM_WARPS == 0);
  uint32_t k_idx = k_idx_base + warp_idx * 4 + lane_idx / 16;  // row idx
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
      cp_async::load_64b_pred(frag[i][j], *gptr, k_idx < kv_len);
      *gptr += 16 * upcast_size_64b<DTypeKV>();
    }
    k_idx += NUM_WARPS * 4;
    *gptr += NUM_WARPS * 4 * stride_n - 4 * NUM_MMA_D * upcast_size_64b<DTypeKV>();
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_k_w(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    uint32_t (*frag)[KTraits::NUM_MMA_D_QK / (8 / sizeof(typename KTraits::DTypeKV))][4]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_K;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    // using swizzle pattern <3, 3, 3>
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
        smem.store_128b(*smem_offset, frag[i][j]);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
      }
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DTypeKV) * NUM_MMA_D;
    }
    *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_k_w_64b(smem_t<KTraits::SWIZZLE_MODE_KV> smem,
                                                uint32_t* smem_offset,
                                                uint32_t (*frag)[KTraits::NUM_MMA_D_QK / 4][2]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_K_64B;
  static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
      smem.store_64b(*smem_offset, frag[i][j]);
      *smem_offset = smem.template advance_offset_by_column<16>(*smem_offset, j);
    }
    *smem_offset = smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
                   4 * NUM_MMA_D;
  }
  *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
}

template <typename KTraits>
__device__ __forceinline__ void produce_k_w_64b(uint64_t* (*k_smem_w)[KTraits::NUM_MMA_D_QK / 4],
                                                uint32_t (*frag)[KTraits::NUM_MMA_D_QK / 4][2]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
      smem_store_64b(k_smem_w[i][j], frag[i][j]);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_v_r_b128(typename KTraits::DTypeKV** gptr,
                                                 const uint32_t stride_n, const uint32_t v_idx_base,
                                                 const uint32_t kv_len, uint32_t* frag) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t v_idx = v_idx_base + warp_idx * 8 + lane_idx / 8;  // row idx
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
        cp_async::load_128b_pred(&frag[i * NUM_MMA_D / (8 / sizeof(DTypeKV)) * 4 + j * 4], *gptr,
                                 v_idx < kv_len);
        *gptr += 8 * upcast_size<DTypeKV>();
      }
      v_idx += NUM_WARPS * 8;
      *gptr += NUM_WARPS * 8 * stride_n - sizeof(DTypeKV) * NUM_MMA_D * upcast_size<DTypeKV>();
    }
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_v_w_b128(smem_t<KTraits::SWIZZLE_MODE_KV> smem,
                                                 uint32_t* smem_offset, uint32_t* frag) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_V;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DTypeKV)); ++j) {
        smem.store_128b(*smem_offset, &frag[i * NUM_MMA_D / (8 / sizeof(DTypeKV)) * 4 + j * 4]);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
      }
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DTypeKV) * NUM_MMA_D;
    }
    *smem_offset -= CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_v_r_b64x4(typename KTraits::DTypeKV** gptr,
                                                  const uint32_t stride_n,
                                                  const uint32_t v_idx_base, const uint32_t kv_len,
                                                  uint32_t* frag) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    // pattern: ldg 4x4_b16
    if constexpr (NUM_MMA_KV % NUM_WARPS_Q == 0) {
      uint32_t (*v_frag)[NUM_MMA_D / 4][4][2] = (uint32_t (*)[NUM_MMA_D / 4][4][2]) frag;
      uint32_t v_idx = v_idx_base + warp_idx * 16 + lane_idx / 16 * 4;  // row idx
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_KV / NUM_WARPS_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
#pragma unroll
          for (uint32_t k = 0; k < 4; ++k) {
            cp_async::load_64b_pred(v_frag[i][j][k], *gptr, v_idx < kv_len);
            *gptr += stride_n;
            v_idx += 1;
          }
          *gptr = *gptr - stride_n * 4 + 16 * upcast_size_64b<DTypeKV>();
          v_idx -= 4;
        }
        v_idx += NUM_WARPS * 16;
        *gptr += NUM_WARPS * 16 * stride_n - NUM_MMA_D * 4 * upcast_size_64b<DTypeKV>();
      }
    } else {
      uint32_t warp_idx_in_wg = warp_idx % 4;
      uint32_t (*v_frag)[4][2] = (uint32_t (*)[4][2])frag;
      uint32_t v_idx = v_idx_base + warp_idx_in_wg * 16 + lane_idx / 16 * 4;  // row idx
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_D / 8; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          cp_async::load_64b_pred(v_frag[i][j], *gptr, v_idx < kv_len);
          *gptr += stride_n;
          v_idx += 1;
        }
        *gptr = *gptr - stride_n * 4 + 32 * upcast_size_64b<DTypeKV>();
        v_idx -= 4;
      }
      *gptr += NUM_WARPS / 2 * 16 * stride_n - NUM_MMA_D * 4 * upcast_size_64b<DTypeKV>();
    }
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void produce_v_w_b64x4(smem_t<KTraits::SWIZZLE_MODE_KV> smem,
                                                  uint32_t* smem_offset, uint32_t* frag) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_V_64B;
  uint32_t perm_frag[4][2];
  uint32_t v_offset = *smem_offset;

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    if constexpr (NUM_MMA_KV % NUM_WARPS_Q == 0) {
      uint32_t (*v_frag)[NUM_MMA_D / 4][4][2] = (uint32_t (*)[NUM_MMA_D / 4][4][2]) frag;
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_KV / NUM_WARPS_Q; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < NUM_MMA_D / 4; ++j) {
          permute_64bx4(v_frag[i][j], perm_frag);
          smem.store_64b(v_offset + 0, perm_frag[0]);
          smem.store_64b(v_offset + 16, perm_frag[1]);
          smem.store_64b(v_offset + 32, perm_frag[2]);
          smem.store_64b(v_offset + 48, perm_frag[3]);
          v_offset = smem.template advance_offset_by_column<64>(v_offset);
        }
        v_offset = smem.template advance_offset_by_row<NUM_WARPS * 16, UPCAST_STRIDE>(v_offset) -
                   NUM_MMA_D * 16;  // NOTE: NUM_MMA_D / 4 * 64
      }
    } else {
      uint32_t (*v_frag)[4][2] = (uint32_t (*)[4][2])frag;
#pragma unroll
      for (uint32_t i = 0; i < NUM_MMA_D / 8; ++i) {
        permute_64bx4(v_frag[i], perm_frag);
        smem.store_64b(v_offset + 0, perm_frag[0]);
        smem.store_64b(v_offset + 16, perm_frag[1]);
        smem.store_64b(v_offset + 32, perm_frag[2]);
        smem.store_64b(v_offset + 48, perm_frag[3]);
        v_offset = smem.template advance_offset_by_column<128>(v_offset);
      }
    }
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

// for cta_kv_tile=64
template <typename KTraits>
__device__ __forceinline__ void produce_v_w_b64x4(uint64_t* (*v_smem_w)[4], uint32_t* frag) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_V_64B;
  uint32_t perm_frag[4][2];
  uint32_t (*v_frag)[4][2] = (uint32_t (*)[4][2])frag;
  constexpr uint32_t NUM_MMA_D =
      (NUM_MMA_KV % NUM_WARPS_Q == 0) ? NUM_MMA_D_VO / 4 : NUM_MMA_D_VO / 8;

#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_D; ++i) {
    permute_64bx4(v_frag[i], perm_frag);
    smem_store_64b(v_smem_w[i][0], perm_frag[0]);
    smem_store_64b(v_smem_w[i][1], perm_frag[1]);
    smem_store_64b(v_smem_w[i][2], perm_frag[2]);
    smem_store_64b(v_smem_w[i][3], perm_frag[3]);
  }
}

template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t kv_idx_base, const size_t* thr_local_kv_offset, const uint32_t kv_len) {
  // NOTE: for fp8, this function doesn't work for head_dim = 64 at the moment
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr SharedMemFillMode fill_mode =
      produce_v ? SharedMemFillMode::kFillZero : SharedMemFillMode::kNoFill;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = produce_v ? KTraits::NUM_MMA_D_VO : KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE =
      produce_v ? KTraits::UPCAST_STRIDE_V : KTraits::UPCAST_STRIDE_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;
  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t kv_idx = kv_idx_base + warp_idx * 4 + lane_idx / 8;
    // NOTE: NUM_MMA_KV * 4 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 4 / num_warps
    static_assert(NUM_MMA_KV * 4 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * upcast_size<DType>();
      }
      kv_idx += NUM_WARPS * 4;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 4, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    uint32_t kv_idx = kv_idx_base + warp_idx * 8 + lane_idx / 4;
    // NOTE: NUM_MMA_KV * 2 / NUM_WARPS_Q = NUM_WARPS_KV * NUM_MMA_KV * 2 / num_warps
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = produce_v ? paged_kv.v_data + thr_local_kv_offset[i]
                              : paged_kv.k_data + thr_local_kv_offset[i];
      smem.load_128b_async<fill_mode>(*smem_offset, gptr, kv_idx < kv_len);
      kv_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset);
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  }
}

template <typename KTraits>
__device__ __forceinline__ void page_produce_k(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t k_idx_base, const size_t* thr_local_k_offset, const uint32_t kv_len) {
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_K;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;
  uint32_t frag[4];

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t k_idx = k_idx_base + warp_idx * 8 + lane_idx / 8;
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = paged_kv.k_data + thr_local_k_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        cp_async::load_128b_pred(frag, gptr, k_idx < kv_len);
        smem.store_128b(*smem_offset, frag);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * upcast_size<DType>();
      }
      k_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D;
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void page_produce_v(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t v_idx_base, const size_t* thr_local_v_offset, const uint32_t kv_len) {
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_V;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;
  uint32_t frag[4];

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t v_idx = v_idx_base + warp_idx * 8 + lane_idx / 8;
    static_assert(NUM_MMA_KV * 2 % NUM_WARPS_Q == 0);
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * 2 / NUM_WARPS_Q; ++i) {
      DType* gptr = paged_kv.v_data + thr_local_v_offset[i];
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        cp_async::load_128b_pred(frag, gptr, v_idx < kv_len);
        smem.store_128b(*smem_offset, frag);
        *smem_offset = smem.template advance_offset_by_column<8>(*smem_offset, j);
        gptr += 8 * upcast_size<DType>();
      }
      v_idx += NUM_WARPS * 8;
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 8, UPCAST_STRIDE>(*smem_offset) -
          NUM_MMA_D * sizeof(DType);
    }
    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void page_produce_v(
    smem_t<KTraits::SWIZZLE_MODE_KV> smem, uint32_t* smem_offset,
    const paged_kv_t<typename KTraits::DTypeKV, typename KTraits::IdType>& paged_kv,
    const uint32_t v_idx_base, const size_t (*thr_local_v_offset)[4], const uint32_t kv_len) {
  using DType = typename KTraits::DTypeKV;
  using IdType = typename KTraits::IdType;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t UPCAST_STRIDE = KTraits::UPCAST_STRIDE_V;
  const uint32_t warp_idx = get_warp_idx<KTraits>(), lane_idx = threadIdx.x;
  uint32_t frag[NUM_MMA_KV / NUM_WARPS_Q][NUM_MMA_D / (8 / sizeof(DType))][4][2];
  uint32_t perm_frag[4][2];

  if constexpr (KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B) {
    uint32_t v_idx = v_idx_base + warp_idx * 16 + lane_idx / 16 * 4;
    // static_assert(NUM_MMA_KV % NUM_WARPS_Q == 0);

#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        DType* gptr = paged_kv.v_data + thr_local_v_offset[i][j];
#pragma unroll
        for (uint32_t k = 0; k < NUM_MMA_D / (8 / sizeof(DType)); ++k) {
          cp_async::load_64b_pred(frag[i][k][j], gptr, v_idx < kv_len);
          gptr += 16 * upcast_size_64b<DType>();
        }
        v_idx += 1;
      }
      v_idx += (NUM_WARPS * 16 - 4);
    }

#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV / NUM_WARPS_Q; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < NUM_MMA_D / (8 / sizeof(DType)); ++j) {
        permute_64bx4(frag[i][j], perm_frag);
        smem.store_128b(*smem_offset, perm_frag[0]);
        smem.store_128b((*smem_offset) + 1, perm_frag[2]);
        *smem_offset = smem.template advance_offset_by_column<32>(*smem_offset, j);
      }
      *smem_offset =
          smem.template advance_offset_by_row<NUM_WARPS * 16, UPCAST_STRIDE>(*smem_offset) -
          sizeof(DType) * NUM_MMA_D * 4;
    }

    *smem_offset -= KTraits::CTA_TILE_KV * UPCAST_STRIDE;
  } else {
    static_assert("SwizzleMode::k64B is not supported");
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_rope_freq(float (*rope_freq)[4], const float rope_rcp_scale,
                                               const float rope_rcp_theta) {
  constexpr uint32_t HEAD_DIM = KTraits::NUM_MMA_D_QK * 16;
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 2; ++mma_d) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      rope_freq[mma_d][j] =
          rope_rcp_scale *
          __powf(rope_rcp_theta,
                 float(2 * ((mma_d * 16 + (j / 2) * 8 + (lane_idx % 4) * 2 + (j % 2)) %
                            (HEAD_DIM / 2))) /
                     float(HEAD_DIM));
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void init_states(typename KTraits::AttentionVariant variant,
                                            float (*o_frag)[KTraits::NUM_MMA_D_VO][4],
                                            typename KTraits::DTypeQKAccum* m, float* d) {
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        o_frag[mma_q][mma_d][reg_id] = 0.f;
      }
    }
  }

  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      m[mma_q] = typename KTraits::DTypeQKAccum(-math::inf);
      d[mma_q] = 1.f;
    }
  }
}

// if use ldg_bsm, we need to swizzle the gmem data
template <typename KTraits, bool USE_LDGBSM = false>
__device__ __forceinline__ void load_q_global_smem(
    uint32_t packed_offset, const uint32_t qo_upper_bound, typename KTraits::DTypeQ* q_ptr_base,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  const uint32_t lane_idx = threadIdx.x, warp_idx_x = get_warp_idx_q<KTraits>();

  if constexpr (USE_LDGBSM) {
    uint32_t q_smem_offset_w = (warp_idx_x * KTraits::NUM_MMA_Q * 16) * UPCAST_STRIDE_Q + lane_idx;

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t row_idx = lane_idx / 8 + mma_q * 16 + j * 8;
        uint32_t q, r;
        group_size.divmod(packed_offset + row_idx, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h;
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
          uint32_t q_offset_r = cp_async::get_permuted_offset(row_idx, mma_do * 8 + lane_idx % 8) *
                                upcast_size<DTypeQ>();
          // load q fragment from gmem to smem
          q_smem->template load_128b_async<DTypeQ, false>(q_smem_offset_w, q_ptr + q_offset_r,
                                                          q_idx < qo_upper_bound);
          q_smem_offset_w += 64;
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<8, UPCAST_STRIDE_Q>(q_smem_offset_w) -
            16 * KTraits::NUM_MMA_D_QK;
      }
    }
  } else {
    uint32_t frag[4];

    uint32_t q_smem_offset_w = q_smem->template get_permuted_offset<UPCAST_STRIDE_Q>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + lane_idx / 8, lane_idx % 8);

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q, r;
        group_size.divmod(packed_offset + lane_idx / 8 + mma_q * 16 + j * 8, q, r);
        const uint32_t q_idx = q;
        DTypeQ* q_ptr =
            q_ptr_base + q * q_stride_n + r * q_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();
#pragma unroll
        for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
          // load q fragment from gmem to reg, then to smem with swizzle
          cp_async::load_128b_pred(frag, q_ptr, q_idx < qo_upper_bound);
          q_smem->store_128b(q_smem_offset_w, frag);
          q_smem_offset_w = q_smem->template advance_offset_by_column<8>(q_smem_offset_w, mma_do);
          q_ptr += 8 * upcast_size<DTypeQ>();
        }
        q_smem_offset_w =
            q_smem->template advance_offset_by_row<8, UPCAST_STRIDE_Q>(q_smem_offset_w) -
            2 * KTraits::NUM_MMA_D_QK;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q_global_smem_64b(
    uint32_t packed_offset, const uint32_t qo_upper_bound, typename KTraits::DTypeQ* q_ptr_base,
    const uint32_t q_stride_n, const uint32_t q_stride_h, const uint_fastdiv group_size,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q_64B;
  const uint32_t lane_idx = threadIdx.x, warp_idx_x = get_warp_idx_q<KTraits>();
  uint32_t frag[2];
  uint32_t q_smem_offset_w[4];
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) {
    q_smem_offset_w[i] = q_smem->template get_permuted_offset_64b<UPCAST_STRIDE_Q>(
        warp_idx_x * KTraits::NUM_MMA_Q * 16 + i * 4 + lane_idx / 16, lane_idx % 16);
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t q, r;
      group_size.divmod(packed_offset + lane_idx / 16 + mma_q * 16 + i * 4, q, r);
      const uint32_t q_idx = q;
      DTypeQ* q_ptr = q_ptr_base + q * q_stride_n + r * q_stride_h +
                      (lane_idx % 16) * upcast_size_64b<DTypeQ>();
      uint32_t q_smem_offset = q_smem_offset_w[i];
#pragma unroll
      for (uint32_t mma_do = 0; mma_do < KTraits::NUM_MMA_D_QK / 4; ++mma_do) {
        // load q fragment from gmem to reg, then to smem with swizzle
        cp_async::load_64b_pred(frag, q_ptr, q_idx < qo_upper_bound);
        q_smem->store_64b(q_smem_offset, frag);
        q_smem_offset = q_smem->template advance_offset_by_column<16>(q_smem_offset);
        q_ptr += 16 * upcast_size_64b<DTypeQ>();
      }
    }
  }
}

template <typename KTraits, bool USE_LDGBSM = false>
__device__ __forceinline__ void load_q_smem_reg(smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem,
                                                uint32_t* q_smem_offset_r,
                                                uint32_t (*q_frag)[KTraits::NUM_MMA_D_QK / 2][4]) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;

  if constexpr (USE_LDGBSM) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          uint32_t* frag = &q_frag[mma_q][mma_d * 2 + j][0];
          q_smem->load_128b(q_smem_offset_r[j], frag);
          q_smem_offset_r[j] =
              q_smem->template advance_offset_by_row<16, UPCAST_STRIDE_Q>(q_smem_offset_r[j]);
        }
      }

#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        q_smem_offset_r[j] = q_smem_offset_r[j] + 64 - KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 2; ++mma_d) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
        uint32_t* frag = &q_frag[mma_q][mma_d][0];
        q_smem->load_128b(*q_smem_offset_r, frag);
        *q_smem_offset_r =
            q_smem->template advance_offset_by_row<16, UPCAST_STRIDE_Q>(*q_smem_offset_r);
      }
      *q_smem_offset_r = q_smem->template advance_offset_by_column<4>(*q_smem_offset_r, mma_d) -
                         NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void load_q_smem_reg_64b(smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem,
                                                    uint32_t* q_smem_offset_r,
                                                    uint32_t (*q_frag)[KTraits::NUM_MMA_D_QK][2]) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_D = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q_64B;

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
#pragma unroll
    for (uint32_t d = 0; d < 4; ++d) {
      q_smem->load_64b(q_smem_offset_r[d], q_frag[0][mma_d * 4 + d]);
      q_smem_offset_r[d] = q_smem->template advance_offset_by_column<16>(q_smem_offset_r[d]);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary(
    const uint32_t q_packed_idx, const uint32_t qo_len, const uint32_t kv_len,
    const uint_fastdiv group_size, smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4]) {
  if (get_warp_idx_kv<KTraits>() == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope<typename KTraits::DTypeQ>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di],
            q_packed_idx + kv_len * group_size - qo_len * group_size + mma_q * 16 + lane_idx / 4,
            group_size);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void q_smem_inplace_apply_rotary_with_pos(
    const uint32_t q_packed_idx_base, const typename KTraits::IdType* q_rope_offset,
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, const uint_fastdiv group_size,
    uint32_t* q_smem_offset_r, float (*rope_freq)[4]) {
  if (get_warp_idx_kv<KTraits>() == 0) {
    constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
    const uint32_t lane_idx = threadIdx.x;
    uint32_t q_frag_local[2][4];
    static_assert(KTraits::NUM_MMA_D_QK % 4 == 0, "NUM_MMA_D_QK must be a multiple of 4");
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      uint32_t q_smem_offset_r_first_half = *q_smem_offset_r;
#pragma unroll
      for (uint32_t mma_di = 0; mma_di < KTraits::NUM_MMA_D_QK / 2; ++mma_di) {
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        uint32_t q_smem_offset_r_last_half =
            q_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                q_smem_offset_r_first_half, 0);
        q_smem->ldmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_frag_apply_llama_rope_with_pos<typename KTraits::DTypeQ, typename KTraits::IdType>(
            (typename KTraits::DTypeQ*)q_frag_local[0], (typename KTraits::DTypeQ*)q_frag_local[1],
            rope_freq[mma_di], q_packed_idx_base + mma_q * 16 + lane_idx / 4, group_size,
            q_rope_offset);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_last_half, q_frag_local[1]);
        q_smem->stmatrix_m8n8x4(q_smem_offset_r_first_half, q_frag_local[0]);
        q_smem_offset_r_first_half =
            q_smem->template advance_offset_by_column<2>(q_smem_offset_r_first_half, mma_di);
      }
      *q_smem_offset_r += 16 * UPCAST_STRIDE_Q;
    }
    *q_smem_offset_r -= KTraits::NUM_MMA_Q * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,
    float (*rope_freq)[4]) {
  using DTypeKV = typename KTraits::DTypeKV;
  static_assert(sizeof(DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  uint32_t k_frag_local[2][4];
  const uint32_t lane_idx = threadIdx.x;
  if constexpr (KTraits::NUM_MMA_D_QK == 4 && KTraits::NUM_WARPS_Q == 4) {
    static_assert(KTraits::NUM_WARPS_KV == 1);
    const uint32_t warp_idx = get_warp_idx_q<KTraits>();
    // horizontal-axis: y
    // vertical-axis: z
    //         | 1-16       | 16-32      | 32-48      | 48-64      |
    // | 1-16  | warp_idx=0 | warp_idx=1 | warp_idx=0 | warp_idx=1 |
    // | 16-32 | warp_idx=2 | warp_idx=3 | warp_idx=2 | warp_idx=3 |
    static_assert(KTraits::NUM_MMA_KV % 2 == 0,
                  "when NUM_MMA_D_QK == 4, NUM_MMA_KV must be a multiple of 2");
    uint32_t kv_idx = kv_idx_base + (warp_idx / 2) * 16 + lane_idx / 4;
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) + (warp_idx / 2) * 16 * UPCAST_STRIDE_K;
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV / 2; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
      uint32_t mma_di = (warp_idx % 2);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      uint32_t k_smem_offset_r_last_half =
          k_smem->template advance_offset_by_column<4>(k_smem_offset_r_first_half, 0);
      k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                       rope_freq[mma_di], kv_idx);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
      k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
      *k_smem_offset_r += 32 * UPCAST_STRIDE_K;
      kv_idx += 32;
    }
    *k_smem_offset_r = (*k_smem_offset_r ^ (0x2 * (warp_idx % 2))) -
                       ((warp_idx / 2) + KTraits::NUM_MMA_KV) * 16 * UPCAST_STRIDE_K;
  } else {
    const uint32_t warp_idx_x = get_warp_idx_q<KTraits>(), warp_idx_z = get_warp_idx_kv<KTraits>();
    static_assert(KTraits::NUM_MMA_D_QK % (2 * KTraits::NUM_WARPS_Q) == 0);
    // horizontal axis: y
    // vertical axis: z
    // | (warp_idx_z, warp_idx_x)       | 1-16   | 16-32  | 32-48  | 48-64  | ...
    // | 1-16*NUM_MMA_KV                | (0, 0) | (0, 1) | (0, 2) | (0, 3) | ...
    // | 16*NUM_MMA_KV-32*NUM_MMA_KV    | (1, 0) | (1, 1) | (1, 2) | (1, 3) | ...
    // ...
    uint32_t kv_idx = kv_idx_base + (warp_idx_z * KTraits::NUM_MMA_KV * 16) + lane_idx / 4;
    *k_smem_offset_r = *k_smem_offset_r ^ (0x2 * warp_idx_x);
#pragma unroll
    for (uint32_t i = 0; i < KTraits::NUM_MMA_KV; ++i) {
      uint32_t k_smem_offset_r_first_half = *k_smem_offset_r;
#pragma unroll
      for (uint32_t j = 0; j < KTraits::NUM_MMA_D_QK / (2 * KTraits::NUM_WARPS_Q); ++j) {
        uint32_t mma_di = warp_idx_x + j * KTraits::NUM_WARPS_Q;
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        uint32_t k_smem_offset_r_last_half =
            k_smem->template advance_offset_by_column<KTraits::NUM_MMA_D_QK>(
                k_smem_offset_r_first_half, 0);
        k_smem->ldmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_frag_apply_llama_rope<DTypeKV>((DTypeKV*)k_frag_local[0], (DTypeKV*)k_frag_local[1],
                                         rope_freq[mma_di], kv_idx);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_last_half, k_frag_local[1]);
        k_smem->stmatrix_m8n8x4(k_smem_offset_r_first_half, k_frag_local[0]);
        k_smem_offset_r_first_half =
            k_smem->template advance_offset_by_column<2 * KTraits::NUM_WARPS_Q>(
                k_smem_offset_r_first_half, mma_di);
      }
      *k_smem_offset_r += 16 * UPCAST_STRIDE_K;
      kv_idx += 16;
    }
    *k_smem_offset_r =
        (*k_smem_offset_r ^ (0x2 * warp_idx_x)) - KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
  }
}

// for lds_b128 & ldgbsm
template <typename KTraits, bool USE_LDGBSM = false>
__device__ __forceinline__ void compute_qk(
    uint32_t (*q_frag)[KTraits::NUM_MMA_D_QK / 2][4], smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem,
    uint32_t* k_smem_offset_r, typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4]) {
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  uint32_t k_frag[4];

  if constexpr (USE_LDGBSM) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK / 4; ++mma_d) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
#pragma unroll
        for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
          k_smem->load_128b(k_smem_offset_r[j], k_frag);
          k_smem_offset_r[j] =
              k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(k_smem_offset_r[j]);

#pragma unroll
          for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                s_frag[mma_q][mma_kv], q_frag[mma_q][mma_d * 2 + j], k_frag);
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                s_frag[mma_q][mma_kv], q_frag[mma_q][mma_d * 2 + j] + 2, k_frag + 2);
          }
        }
        k_smem_offset_r[j] -= KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
      }

#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        k_smem_offset_r[j] += 64;
      }
    }

#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      k_smem_offset_r[j] -= KTraits::NUM_MMA_D_QK * 16;
    }
  } else {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK / 2; ++mma_d) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        k_smem->load_128b(*k_smem_offset_r, k_frag);
        *k_smem_offset_r =
            k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(*k_smem_offset_r);

#pragma unroll
        for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              s_frag[mma_q][mma_kv], q_frag[mma_q][mma_d], k_frag);
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              s_frag[mma_q][mma_kv], q_frag[mma_q][mma_d] + 2, k_frag + 2);
        }
      }
      *k_smem_offset_r = k_smem->template advance_offset_by_column<4>(*k_smem_offset_r, mma_d) -
                         KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    }
    *k_smem_offset_r -= KTraits::NUM_MMA_D_QK * sizeof(typename KTraits::DTypeKV);
  }
}

// for lds_b64
template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    uint32_t (*q_frag)[KTraits::NUM_MMA_D_QK][2], smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem,
    uint32_t* k_smem_offset_r, typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4]) {
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K_64B;

  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK / 4; ++mma_d) {
#pragma unroll
    for (uint32_t d = 0; d < 4; ++d) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        uint32_t k_frag[2];
        k_smem->load_64b(k_smem_offset_r[d], k_frag);
        k_smem_offset_r[d] =
            k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(k_smem_offset_r[d]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[0][mma_kv], q_frag[0][mma_d * 4 + d], k_frag);
      }
      k_smem_offset_r[d] = k_smem->template advance_offset_by_column<16>(k_smem_offset_r[d]) -
                           KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
    }
  }

#pragma unroll
  for (uint32_t d = 0; d < 4; ++d) {
    k_smem_offset_r[d] -= KTraits::NUM_MMA_D_QK * 4;
  }
}

template <typename KTraits>
__device__ __forceinline__ void calculate_smem_ptr_r(
    smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint64_t* (*k_smem_ptr_r)[4][KTraits::NUM_MMA_KV],
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem,
    uint64_t* (*v_smem_ptr_r)[KTraits::NUM_MMA_D_VO / 4][4]) {
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_K_64B = KTraits::UPCAST_STRIDE_K_64B;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;
  constexpr uint32_t V_THR_LAYOUT_COL = KTraits::V_THR_LAYOUT_COL;
  constexpr uint32_t V_THR_LAYOUT_ROW = KTraits::V_THR_LAYOUT_ROW;
  constexpr uint32_t NUM_WARPS = KTraits::NUM_WARPS;
  constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<KTraits>();

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK / 4; ++mma_d) {
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      uint32_t offset =
          k_smem->template get_permuted_offset_64b<UPCAST_STRIDE_K_64B>(
              get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + lane_idx % 16, 4 * i + lane_idx / 16) +
          mma_d * 16;
      k_smem_ptr_r[mma_d][i][0] = offset + (uint64_t*)k_smem->base;
#pragma unroll
      for (uint32_t mma_kv = 1; mma_kv < NUM_MMA_KV; ++mma_kv) {
        offset = k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K_64B>(offset);
        k_smem_ptr_r[mma_d][i][mma_kv] = offset + (uint64_t*)k_smem->base;
      }
    }
  }

  if constexpr (NUM_MMA_D_VO == 8) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t offset = v_smem->template get_64bx4_offset<UPCAST_STRIDE_V_64B>(
                            lane_idx / V_THR_LAYOUT_COL, lane_idx % V_THR_LAYOUT_COL) +
                        16 * UPCAST_STRIDE_V_64B * mma_kv;
      v_smem_ptr_r[mma_kv][0][0] = offset + (uint64_t*)v_smem->base;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO / 4; ++mma_d) {
        offset = offset + 64 * mma_d;
        v_smem_ptr_r[mma_kv][mma_d][0] = offset + (uint64_t*)v_smem->base;
#pragma unroll
        for (uint32_t c = 1; c < 4; ++c) {
          v_smem_ptr_r[mma_kv][mma_d][c] = offset + 16 * c + (uint64_t*)v_smem->base;
        }
      }
    }
  } else {
    uint32_t base_offset = v_smem->template get_64bx4_offset<UPCAST_STRIDE_V_64B>(
        lane_idx / V_THR_LAYOUT_COL, lane_idx % V_THR_LAYOUT_COL);
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO / 4; ++mma_d) {
        uint32_t offset = base_offset + 64 * mma_d + 16 * UPCAST_STRIDE_V_64B * mma_kv;
        v_smem_ptr_r[mma_kv][mma_d][0] = offset + (uint64_t*)v_smem->base;
#pragma unroll
        for (uint32_t c = 1; c < 4; ++c) {
          v_smem_ptr_r[mma_kv][mma_d][c] = offset + 16 * c + (uint64_t*)v_smem->base;
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    uint32_t (*q_frag)[KTraits::NUM_MMA_D_QK][2], uint64_t* (*k_smem_ptr_r)[4][KTraits::NUM_MMA_KV],
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4]) {
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K_64B;

  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK / 4; ++mma_d) {
#pragma unroll
    for (uint32_t d = 0; d < 4; ++d) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        uint32_t k_frag[2];
        smem_load_64b(k_smem_ptr_r[mma_d][d][mma_kv], k_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[0][mma_kv], q_frag[0][mma_d * 4 + d], k_frag);
      }
    }
  }
}

// for prefetch lds_k
template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    uint32_t (*q_frag)[KTraits::NUM_MMA_D_QK / 2][4], uint32_t (*k_frag)[KTraits::NUM_MMA_KV][4],
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4]) {
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK / 2; ++mma_d) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[mma_q][mma_kv], q_frag[mma_q][mma_d], k_frag[mma_d][mma_kv]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[mma_q][mma_kv], q_frag[mma_q][mma_d] + 2, k_frag[mma_d][mma_kv] + 2);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void lds_k(smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem,
                                      uint32_t* k_smem_offset_r,
                                      uint32_t (*k_frag)[KTraits::NUM_MMA_KV][4]) {
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_QK / 2; ++mma_d) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      k_smem->load_128b(*k_smem_offset_r, k_frag[mma_d][mma_kv]);
      *k_smem_offset_r =
          k_smem->template advance_offset_by_row<16, UPCAST_STRIDE_K>(*k_smem_offset_r);
    }
    *k_smem_offset_r = k_smem->template advance_offset_by_column<4>(*k_smem_offset_r, mma_d) -
                       KTraits::NUM_MMA_KV * 16 * UPCAST_STRIDE_K;
  }
  *k_smem_offset_r -= KTraits::NUM_MMA_D_QK * sizeof(typename KTraits::DTypeKV);
}

template <typename KTraits, typename Params, typename DTypeQKAccum>
__device__ __forceinline__ void logits_transform(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint_fastdiv group_size,
    DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4], const uint32_t kv_head_idx) {
  const uint32_t lane_idx = threadIdx.x;
  uint32_t q[KTraits::NUM_MMA_Q], r[KTraits::NUM_MMA_Q];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
    group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx % 16, q[mma_q], r[mma_q]);
  }
  uint32_t qo_head_idx = kv_head_idx * group_size;
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
    const uint32_t q_idx = q[mma_q];
    qo_head_idx += r[mma_q];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      uint32_t kv_idx = kv_idx_base + mma_kv * 16 + lane_idx / 16 * 4;
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        kv_idx += reg_id;
        s_frag[mma_q][mma_kv][reg_id] =
            variant.LogitsTransform(params, s_frag[mma_q][mma_kv][reg_id], batch_idx, q_idx, kv_idx,
                                    qo_head_idx, kv_head_idx);
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void logits_mask(
    const Params& params, typename KTraits::AttentionVariant variant, const uint32_t batch_idx,
    const uint32_t qo_packed_idx_base, const uint32_t kv_idx_base, const uint32_t qo_len,
    const uint32_t kv_len, const uint32_t chunk_end, const uint_fastdiv group_size,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4], const uint32_t kv_head_idx) {
  const uint32_t lane_idx = threadIdx.x;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
  uint32_t q[NUM_MMA_Q], r[NUM_MMA_Q];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
    group_size.divmod(qo_packed_idx_base + mma_q * 16 + lane_idx % 16, q[mma_q], r[mma_q]);
  }
  uint32_t qo_head_idx = kv_head_idx * group_size;
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
    const uint32_t q_idx = q[mma_q];
    qo_head_idx += r[mma_q];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t kv_idx_star = kv_idx_base + mma_kv * 16 + lane_idx / 16 * 4;
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        const uint32_t kv_idx = kv_idx_star + (reg_id % 4);
        const bool mask =
            (!(MASK_MODE == MaskMode::kCausal
                   ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= chunk_end))
                   : kv_idx >= chunk_end)) &&
            variant.LogitsMask(params, batch_idx, q_idx, kv_idx, qo_head_idx, kv_head_idx);
        s_frag[mma_q][mma_kv][reg_id] =
            (mask) ? s_frag[mma_q][mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states(
    typename KTraits::AttentionVariant variant,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], typename KTraits::DTypeQKAccum* m, float* d) {
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  constexpr bool use_softmax = AttentionVariant::use_softmax;
  if constexpr (use_softmax) {
    const float sm_scale = variant.sm_scale_log2;
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      float m_prev = m[mma_q];
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        float m_local = max(max(s_frag[mma_q][mma_kv][0], s_frag[mma_q][mma_kv][1]),
                            max(s_frag[mma_q][mma_kv][2], s_frag[mma_q][mma_kv][3]));
        m[mma_q] = max(m[mma_q], m_local);
      }

      m[mma_q] = max(m[mma_q], math::shfl_xor_sync(m[mma_q], 32));
      m[mma_q] = max(m[mma_q], math::shfl_xor_sync(m[mma_q], 16));

      float o_scale = math::ptx_exp2(m_prev * sm_scale - m[mma_q] * sm_scale);
      d[mma_q] *= o_scale;
      auto m_scale = m[mma_q] * sm_scale * -1;
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        fma_f32x2(&o_frag[mma_q][mma_d][0], &o_frag[mma_q][mma_d][0], o_scale);
        fma_f32x2(&o_frag[mma_q][mma_d][2], &o_frag[mma_q][mma_d][2], o_scale);
      }
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        // s_frag = exp(s_frag * sm_scale - m * sm_scale)
        fma_f32x2(&s_frag[mma_q][mma_kv][0], &s_frag[mma_q][mma_kv][0], sm_scale, m_scale);
        fma_f32x2(&s_frag[mma_q][mma_kv][2], &s_frag[mma_q][mma_kv][2], sm_scale, m_scale);
        s_frag[mma_q][mma_kv][0] = math::ptx_exp2(s_frag[mma_q][mma_kv][0]);
        s_frag[mma_q][mma_kv][1] = math::ptx_exp2(s_frag[mma_q][mma_kv][1]);
        s_frag[mma_q][mma_kv][2] = math::ptx_exp2(s_frag[mma_q][mma_kv][2]);
        s_frag[mma_q][mma_kv][3] = math::ptx_exp2(s_frag[mma_q][mma_kv][3]);
      }
    }
  }
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false, bool USE_LDGBSM = false>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], float* d) {
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][4];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeQ, float>::template cast<4>(s_frag_f16[mma_q][mma_kv],
                                                                  s_frag[mma_q][mma_kv]);
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        mma::m16k16_rowsum_f16f16f32(&d[mma_q], s_frag_f16[mma_q][mma_kv]);
      }
    }
  }

  if constexpr (LDS_TRANS_ENABLE && USE_LDGBSM) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
        uint32_t b_frag[4][2];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          v_smem->load_64b_trans(v_smem_offset_r[i], b_frag[i]);
        }

#pragma unroll
        for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                o_frag[mma_q][mma_d * 4 + i], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[i]);
          }
        }

#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          v_smem_offset_r[i] += 128;
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        v_smem_offset_r[i] =
            v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V_64B>(v_smem_offset_r[i]) -
            32 * KTraits::NUM_MMA_D_VO;
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      v_smem_offset_r[i] -= 16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V_64B;
    }
  } else if (LDS_TRANS_ENABLE && !USE_LDGBSM) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
        uint32_t b_frag[4][2];
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          v_smem->load_64b_trans(v_smem_offset_r[i], b_frag[i]);
        }

#pragma unroll
        for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                o_frag[mma_q][mma_d * 4 + i], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[i]);
          }
        }

#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) {
          v_smem_offset_r[i] =
              v_smem->template advance_offset_by_column<16>(v_smem_offset_r[i], mma_d);
        }
      }

#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        v_smem_offset_r[i] =
            v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V_64B>(v_smem_offset_r[i]) -
            16 * KTraits::NUM_MMA_D_VO / 4;
      }
    }

#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) {
      v_smem_offset_r[i] -= 16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V_64B;
    }
  } else {
    uint32_t v_frag[2];

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
#pragma unroll
        for (uint32_t c = 0; c < 4; ++c) {
          v_smem->load_64b(*v_smem_offset_r + 16 * c, v_frag);
#pragma unroll
          for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
                o_frag[mma_q][mma_d * 4 + c], (uint32_t*)s_frag_f16[mma_q][mma_kv], v_frag);
          }
        }
        *v_smem_offset_r = v_smem->template advance_offset_by_column<64>(*v_smem_offset_r);
      }
      *v_smem_offset_r =
          v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V_64B>(*v_smem_offset_r) -
          16 * KTraits::NUM_MMA_D_VO;  // NOTE: NUM_MMA_D_VO / 4 * 64
    }
    *v_smem_offset_r -= 16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V_64B;
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    uint64_t* (*v_smem_ptr_r)[KTraits::NUM_MMA_D_VO / 4][4],
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], float* d) {
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][4];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeQ, float>::template cast<4>(s_frag_f16[mma_q][mma_kv],
                                                                  s_frag[mma_q][mma_kv]);
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        mma::m16k16_rowsum_f16f16f32(&d[mma_q], s_frag_f16[mma_q][mma_kv]);
      }
    }
  }

  uint32_t v_frag[2];

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
#pragma unroll
      for (uint32_t c = 0; c < 4; ++c) {
        smem_load_64b(v_smem_ptr_r[mma_kv][mma_d][c], v_frag);
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
              o_frag[mma_q][mma_d * 4 + c], (uint32_t*)s_frag_f16[mma_q][mma_kv], v_frag);
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v_with_perm(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r,
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], float* d) {
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][4];
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeQ, float>::template cast<4>(s_frag_f16[mma_q][mma_kv],
                                                                  s_frag[mma_q][mma_kv]);
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        mma::m16k16_rowsum_f16f16f32(&d[mma_q], s_frag_f16[mma_q][mma_kv]);
      }
    }
  }

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
      uint32_t v_frag[4][2];
      uint32_t b_frag[4][2];
      for (int i = 0; i < 4; ++i)  // 4*4 perm
      {
        v_smem->load_64b(*v_smem_offset_r, v_frag[i]);
        *v_smem_offset_r =
            v_smem->template advance_offset_by_row<1, UPCAST_STRIDE_V_64B>(*v_smem_offset_r);
      }
      permute_64bx4(v_frag, b_frag);
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 0], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[0]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 1], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[1]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 2], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[2]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 3], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[3]);
      }
      *v_smem_offset_r = v_smem->template advance_offset_by_column<16>(*v_smem_offset_r, mma_d) -
                         4 * UPCAST_STRIDE_V_64B;
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V_64B>(*v_smem_offset_r) - 2 * 16;
  }
  *v_smem_offset_r -= (16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V_64B);
}

// for prefetch lds_v
template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v_with_perm(
    typename KTraits::DTypeQKAccum (*s_frag)[KTraits::NUM_MMA_KV][4],
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], float* d,
    uint32_t (*v_frag)[KTraits::NUM_MMA_D_VO / 4][4][2]) {
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;

  typename KTraits::DTypeQ s_frag_f16[KTraits::NUM_MMA_Q][KTraits::NUM_MMA_KV][4];

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeQ, float>::template cast<4>(s_frag_f16[mma_q][mma_kv],
                                                                  s_frag[mma_q][mma_kv]);
    }
  }

  if constexpr (KTraits::AttentionVariant::use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        mma::m16k16_rowsum_f16f16f32(&d[mma_q], s_frag_f16[mma_q][mma_kv]);
      }
    }
  }

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
      uint32_t b_frag[4][2];
      permute_64bx4(v_frag[mma_kv][mma_d], b_frag);
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 0], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[0]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 1], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[1]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 2], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[2]);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            o_frag[mma_q][mma_d * 4 + 3], (uint32_t*)s_frag_f16[mma_q][mma_kv], b_frag[3]);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void lds_v(smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem,
                                      uint32_t* v_smem_offset_r,
                                      uint32_t (*v_frag)[KTraits::NUM_MMA_D_VO / 4][4][2]) {
  static_assert(std::is_same_v<typename KTraits::DTypeQKAccum, float>);
  static_assert(sizeof(typename KTraits::DTypeKV) == 2);
  constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO / 4; ++mma_d) {
      for (int i = 0; i < 4; ++i)  // 4*4 perm
      {
        v_smem->load_64b(*v_smem_offset_r, v_frag[mma_kv][mma_d][i]);
        *v_smem_offset_r =
            v_smem->template advance_offset_by_row<1, UPCAST_STRIDE_V_64B>(*v_smem_offset_r);
      }
      *v_smem_offset_r = v_smem->template advance_offset_by_column<16>(*v_smem_offset_r, mma_d) -
                         4 * UPCAST_STRIDE_V_64B;
    }
    *v_smem_offset_r =
        v_smem->template advance_offset_by_row<16, UPCAST_STRIDE_V_64B>(*v_smem_offset_r) - 2 * 16;
  }
  *v_smem_offset_r -= (16 * KTraits::NUM_MMA_KV * UPCAST_STRIDE_V_64B);
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d(float (*o_frag)[KTraits::NUM_MMA_D_VO][4],
                                            typename KTraits::DTypeQKAccum* m, float* d) {
  using AttentionVariant = typename KTraits::AttentionVariant;
  if constexpr (AttentionVariant::use_softmax) {
    float d_rcp[KTraits::NUM_MMA_Q];
    // compute reciprocal of d
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      d_rcp[mma_q] =
          (m[mma_q] != typename KTraits::DTypeQKAccum(-math::inf)) ? math::ptx_rcp(d[mma_q]) : 0.f;
    }

#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
        fma_f32x2(&o_frag[mma_q][mma_d][0], &o_frag[mma_q][mma_d][0], d_rcp[mma_q]);
        fma_f32x2(&o_frag[mma_q][mma_d][2], &o_frag[mma_q][mma_d][2], d_rcp[mma_q]);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m(typename KTraits::AttentionVariant variant,
                                           typename KTraits::DTypeQKAccum* m) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
      if (m[mma_q] != typename KTraits::DTypeQKAccum(-math::inf)) {
        m[mma_q] *= variant.sm_scale_log2;
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], smem_t<KTraits::SWIZZLE_MODE_Q>* o_smem,
    typename KTraits::DTypeO* o_ptr_base, const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound, const uint32_t o_stride_n, const uint32_t o_stride_h,
    const uint_fastdiv group_size) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t UPCAST_STRIDE_O_64B = KTraits::UPCAST_STRIDE_O_64B;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>();
  const uint32_t lane_idx = threadIdx.x;
  uint32_t o_frag_f16[2];

  static_assert(sizeof(DTypeO) == 2);
  if constexpr (sizeof(DTypeO) == 4) {
    // #pragma unroll
    //     for (uint32_t mma_q = 0; mma_q < KTraits::NUM_MMA_Q; ++mma_q) {
    // #pragma unroll
    //       for (uint32_t j = 0; j < 2; ++j) {
    //         uint32_t q, r;
    //         group_size.divmod(o_packed_idx_base + lane_idx / 4 + mma_q * 16 + j * 8, q, r);
    //         const uint32_t o_idx = q;
    // #pragma unroll
    //         for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_VO; ++mma_d) {
    //           if (o_idx < qo_upper_bound) {
    //             *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d *
    //             16 +
    //                                        (lane_idx % 4) * 2) =
    //                 *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][j * 2]);
    //             *reinterpret_cast<float2*>(o_ptr_base + q * o_stride_n + r * o_stride_h + mma_d *
    //             16 +
    //                                        8 + (lane_idx % 4) * 2) =
    //                 *reinterpret_cast<float2*>(&o_frag[mma_q][mma_d][4 + j * 2]);
    //           }
    //         }
    //       }
    //     }
  } else {
    if (get_warp_idx_kv<KTraits>() == 0) {
#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO; ++mma_d) {
          vec_cast<DTypeO, float>::template cast<4>((DTypeO*)o_frag_f16, o_frag[mma_q][mma_d]);
          uint32_t o_smem_offset_w = o_smem->template get_permuted_offset<UPCAST_STRIDE_O_64B, 16>(
              (warp_idx_x * NUM_MMA_Q + mma_q) * 16 + lane_idx % 16, mma_d * 4 + lane_idx / 16);
          o_smem->store_64b(o_smem_offset_w, o_frag_f16);
        }
      }

#pragma unroll
      for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          uint32_t o_smem_offset_r = o_smem->template get_permuted_offset<UPCAST_STRIDE_O_64B, 16>(
              warp_idx_x * NUM_MMA_Q * 16 + mma_q * 16 + j * 4 + lane_idx / 16, lane_idx % 16);

          uint32_t q, r;
          group_size.divmod(o_packed_idx_base + lane_idx / 16 + mma_q * 16 + j * 4, q, r);
          const uint32_t o_idx = q;
          DTypeO* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h +
                          (lane_idx % 16) * upcast_size_64b<DTypeO>();
#pragma unroll
          for (uint32_t mma_do = 0; mma_do < NUM_MMA_D_VO / 4; ++mma_do) {
            if (o_idx < qo_upper_bound) {
              o_smem->load_64b(o_smem_offset_r, o_frag_f16);
              cp_async::store_64b_pred(o_frag_f16, o_ptr, true);
            }
            o_ptr += 16 * upcast_size_64b<DTypeO>();
            o_smem_offset_r =
                o_smem->template advance_offset_by_column<16>(o_smem_offset_r, mma_do);
          }
        }
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem_b128(
    float (*o_frag)[KTraits::NUM_MMA_D_VO][4], smem_t<KTraits::SWIZZLE_MODE_Q>* o_smem,
    typename KTraits::DTypeO* o_ptr_base, const uint32_t o_packed_idx_base,
    const uint32_t qo_upper_bound, const uint32_t o_stride_n, const uint32_t o_stride_h,
    const uint_fastdiv group_size) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t UPCAST_STRIDE_O_64B = KTraits::UPCAST_STRIDE_O_64B;
  constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  const uint32_t warp_idx_x = get_warp_idx_q<KTraits>();
  const uint32_t lane_idx = threadIdx.x;
  uint32_t o_frag_f16[4];
  float o_reset[16];
  static_assert(sizeof(DTypeO) == 2);
#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_VO / 4; ++mma_d) {
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
#pragma unroll
        for (uint32_t j = 0; j < 4; ++j) {
          o_reset[i * 4 + j] = o_frag[mma_q][mma_d * 4 + j][i];
        }
      }
      uint32_t o_smem_offset_w =
          ((warp_idx_x * NUM_MMA_Q + mma_q) * 16 + lane_idx % 16) * UPCAST_STRIDE_O +
          (mma_d * 4 + lane_idx / 16) * 2;

      vec_cast<DTypeO, float>::template cast<8>((DTypeO*)o_frag_f16, o_reset);
      o_smem->store_128b(o_smem_offset_w, o_frag_f16);

      vec_cast<DTypeO, float>::template cast<8>((DTypeO*)o_frag_f16, o_reset + 8);
      o_smem->store_128b(o_smem_offset_w + 1, o_frag_f16);
    }
  }

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t o_smem_offset_r =
          (warp_idx_x * NUM_MMA_Q * 16 + mma_q * 16 + j * 4 + lane_idx / 16) * UPCAST_STRIDE_O_64B +
          lane_idx % 16;

      uint32_t q, r;
      group_size.divmod(o_packed_idx_base + lane_idx / 16 + mma_q * 16 + j * 4, q, r);
      const uint32_t o_idx = q;
      DTypeO* o_ptr = o_ptr_base + q * o_stride_n + r * o_stride_h +
                      (lane_idx % 16) * upcast_size_64b<DTypeO>();
#pragma unroll
      for (uint32_t mma_do = 0; mma_do < NUM_MMA_D_VO / 4; ++mma_do) {
        if (o_idx < qo_upper_bound) {
          o_smem->load_64b(o_smem_offset_r, o_frag_f16);
          cp_async::store_64b_pred(o_frag_f16, o_ptr, true);
        }
        o_ptr += 16 * upcast_size_64b<DTypeO>();
        o_smem_offset_r = o_smem->template advance_offset_by_column<16>(o_smem_offset_r, mma_do);
      }
    }
  }
}

}  // namespace

template <typename KTraits>
using write_o_reg_gmem_ptr = void (*)(float (*)[KTraits::NUM_MMA_D_VO][4],
                                      smem_t<KTraits::SWIZZLE_MODE_Q>*, typename KTraits::DTypeO*,
                                      const uint32_t, const uint32_t, const uint32_t,
                                      const uint32_t, const uint_fastdiv);

template <typename KTraits>
using compute_sfm_v_ptr = void (*)(smem_t<KTraits::SWIZZLE_MODE_KV>*, uint32_t*,
                                   typename KTraits::DTypeQKAccum (*)[KTraits::NUM_MMA_KV][4],
                                   float (*)[KTraits::NUM_MMA_D_VO][4], float*);

template <typename KTraits>
using compute_sfm_v_noperm_ptr =
    void (*)(uint64_t* (*)[KTraits::NUM_MMA_D_VO / 4][4],
             typename KTraits::DTypeQKAccum (*)[KTraits::NUM_MMA_KV][4],
             float (*)[KTraits::NUM_MMA_D_VO][4], float*);

template <typename KTraits>
using produce_v_w_ptr = void (*)(smem_t<KTraits::SWIZZLE_MODE_KV>, uint32_t*, uint32_t*);

template <typename KTraits>
using produce_v_w_b64x4_ptr = void (*)(uint64_t* (*)[4], uint32_t*);

template <typename KTraits>
using produce_v_r_ptr = void (*)(typename KTraits::DTypeKV**, const uint32_t, const uint32_t,
                                 const uint32_t, uint32_t*);

// This general template is a sample, please use the specialized ones.
template <const int CTA_KV_TILE, bool UseLdsTrans, typename KTraits>
struct DeviceFunctionSelector {
  static constexpr write_o_reg_gmem_ptr<KTraits> Write_O_Func = write_o_reg_gmem_b128<KTraits>;
  static constexpr compute_sfm_v_ptr<KTraits> Sfm_V_Func = compute_sfm_v<KTraits>;
  static constexpr produce_v_w_ptr<KTraits> Write_V_Func = produce_v_w_b128<KTraits>;
  static constexpr produce_v_r_ptr<KTraits> Read_V_Func = produce_v_r_b128<KTraits>;
};

template <typename KTraits>
struct DeviceFunctionSelector<64, false, KTraits> {
  static constexpr write_o_reg_gmem_ptr<KTraits> Write_O_Func = write_o_reg_gmem_b128<KTraits>;
  static constexpr compute_sfm_v_noperm_ptr<KTraits> Sfm_V_Func = compute_sfm_v<KTraits>;
  static constexpr produce_v_w_b64x4_ptr<KTraits> Write_V_Func = produce_v_w_b64x4<KTraits>;
  static constexpr produce_v_r_ptr<KTraits> Read_V_Func = produce_v_r_b64x4<KTraits>;
};

template <typename KTraits>
struct DeviceFunctionSelector<64, true, KTraits> {
  static constexpr write_o_reg_gmem_ptr<KTraits> Write_O_Func = write_o_reg_gmem<KTraits>;
  static constexpr compute_sfm_v_ptr<KTraits> Sfm_V_Func = compute_sfm_v<KTraits>;
  static constexpr produce_v_w_ptr<KTraits> Write_V_Func = produce_v_w_b128<KTraits>;
  static constexpr produce_v_r_ptr<KTraits> Read_V_Func = produce_v_r_b128<KTraits>;
};

template <typename KTraits>
struct DeviceFunctionSelector<32, false, KTraits> {
  static constexpr write_o_reg_gmem_ptr<KTraits> Write_O_Func = write_o_reg_gmem_b128<KTraits>;
  static constexpr compute_sfm_v_ptr<KTraits> Sfm_V_Func = compute_sfm_v_with_perm<KTraits>;
  static constexpr produce_v_w_ptr<KTraits> Write_V_Func = produce_v_w_b128<KTraits>;
  static constexpr produce_v_r_ptr<KTraits> Read_V_Func = produce_v_r_b128<KTraits>;
};

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_UTILS_CUH_
