/*
 * Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 */
#ifndef FLASHINFER_MLA_FA2_UTILS_128B_CUH_
#define FLASHINFER_MLA_FA2_UTILS_128B_CUH_

#include <cstdint>
#include <sstream>

#include "mla_params.cuh"
#include "prefill_utils.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

namespace mla {

template <typename KTraits>
__device__ __forceinline__ void compute_kv_offset(uint32_t* kv_page_idx, uint32_t* kv_page_offset,
                                                  int64_t* ckv_offset, int64_t* kpe_offset,
                                                  const uint64_t ckv_stride_n,
                                                  const uint64_t ckv_stride_page,
                                                  const uint64_t kpe_stride_n,
                                                  const uint64_t kpe_stride_page) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV_PER_WAVE; ++mma_kv) {
    ckv_offset[mma_kv] = kv_page_idx[mma_kv] * ckv_stride_page +
                         kv_page_offset[mma_kv] * ckv_stride_n +
                         (lane_idx % 8) * upcast_size<DTypeKV>();
    kpe_offset[mma_kv] = kv_page_idx[mma_kv] * kpe_stride_page +
                         kv_page_offset[mma_kv] * kpe_stride_n +
                         (lane_idx % 8) * upcast_size<DTypeKV>();
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void prefetch_kv_indices(
    typename KTraits::DTypeKV* ckv, typename KTraits::DTypeKV* kpe,
    const uint32_t packed_block_iter_base, const uint_fastdiv& block_size,
    const uint32_t packed_kv_bound, typename KTraits::IdType* indices,
    typename KTraits::DTypeKV*(*ckv_base), typename KTraits::DTypeKV*(*kpe_base),
    const uint64_t ckv_stride_n, const uint64_t ckv_stride_page, const uint64_t kpe_stride_n,
    const uint64_t kpe_stride_page) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t kv_page_idx;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV_PER_WAVE; ++mma_kv) {
    uint32_t q, r;
    // The ldg layout in different tile sizes should match with load_kv
    if constexpr (NUM_MMA_KV == 1) {
      if (warpgroup_idx == 0) {
        uint32_t packed_block_iter = packed_block_iter_base + +lane_idx / 8 + warp_idx_in_wg * 8;
        block_size.divmod(packed_block_iter, q, r);
        bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
        cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
      }
    } else if constexpr (CTA_TILE_Q == 64) {
      if (warpgroup_idx == 0) {
        uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + 64 * mma_kv +
                                     warpgroup_idx * 32 + warp_idx_in_wg * 8;
        block_size.divmod(packed_block_iter, q, r);
        bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
        cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
      }
    } else {
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + 32 * mma_kv +
                                   warpgroup_idx * 16 + warp_idx_in_wg * 8;
      block_size.divmod(packed_block_iter, q, r);
      bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
      cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
    }
    ckv_base[mma_kv] = kv_page_idx * ckv_stride_page + r * ckv_stride_n + ckv;
    kpe_base[mma_kv] = kv_page_idx * kpe_stride_page + r * kpe_stride_n + kpe;
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void prefetch_kv_indices(
    const uint32_t packed_block_iter_base, const uint_fastdiv& block_size,
    const uint32_t packed_kv_bound, typename KTraits::IdType* indices, int64_t* ckv_offset,
    int64_t* kpe_offset, const uint64_t ckv_stride_n, const uint64_t ckv_stride_page,
    const uint64_t kpe_stride_n, const uint64_t kpe_stride_page) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t kv_page_idx;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV_PER_WAVE; ++mma_kv) {
    uint32_t q, r;
    // The ldg layout in different tile sizes should match with load_kv
    if constexpr (NUM_MMA_KV == 1) {
      if (warpgroup_idx == 0) {
        uint32_t packed_block_iter = packed_block_iter_base + +lane_idx / 8 + warp_idx_in_wg * 8;
        block_size.divmod(packed_block_iter, q, r);
        bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
        cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
      }
    } else if constexpr (CTA_TILE_Q == 64) {
      if (warpgroup_idx == 0) {
        uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + 64 * mma_kv +
                                     warpgroup_idx * 32 + warp_idx_in_wg * 8;
        block_size.divmod(packed_block_iter, q, r);
        bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
        cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
      }
    } else {
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + 32 * mma_kv +
                                   warpgroup_idx * 16 + warp_idx_in_wg * 8;
      block_size.divmod(packed_block_iter, q, r);
      bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
      cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
    }
    ckv_offset[mma_kv] =
        kv_page_idx * ckv_stride_page + r * ckv_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
    kpe_offset[mma_kv] =
        kv_page_idx * kpe_stride_page + r * kpe_stride_n + (lane_idx % 8) * upcast_size<DTypeKV>();
  }
}

template <typename KTraits, uint32_t Begin, uint32_t End>
__device__ __forceinline__ void load_kv_r_partial(bool row_mask, uint32_t (*frag)[4],
                                                  typename KTraits::DTypeKV* kv_ptr_base,
                                                  int64_t kv_offset) {
  using DTypeKV = typename KTraits::DTypeKV;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  DTypeKV* kv_ptr = kv_ptr_base + kv_offset;
  kv_ptr += Begin * 8 * upcast_size<DTypeKV>();
#pragma unroll
  for (uint32_t mma_d = Begin; mma_d < End; ++mma_d) {
    cp_async::load_128b_pred(frag[mma_d], kv_ptr, row_mask);
    kv_ptr += 8 * upcast_size<DTypeKV>();
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void get_row_mask(const uint32_t packed_kv_bound,
                                             const uint32_t packed_block_iter_base, bool* row_mask,
                                             uint32_t mma_kv_idx = 0) {
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t packed_block_iter;

  if constexpr (KTraits::NUM_MMA_KV == 1) {
    packed_block_iter = packed_block_iter_base + lane_idx / 8 + warp_idx_in_wg * 8;
  } else if constexpr (CTA_TILE_Q == 64) {
    packed_block_iter = packed_block_iter_base + lane_idx / 8 + 64 * mma_kv_idx +
                        warpgroup_idx * 32 + warp_idx_in_wg * 8;
  } else {
    packed_block_iter = packed_block_iter_base + lane_idx / 8 + 32 * mma_kv_idx +
                        warpgroup_idx * 16 + warp_idx_in_wg * 8;
  }
  *row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
}

// The purpose of this function is to load only a portion of kv.
template <typename KTraits, uint32_t NUM_MMA_D, uint32_t Begin, uint32_t End,
          bool Is_even_MN = false>
__device__ __forceinline__ void load_kv_r(typename KTraits::DTypeKV* kv,
                                          uint32_t (*kv_frag)[NUM_MMA_D / 4][4], int64_t* kv_offset,
                                          const uint32_t packed_kv_bound,
                                          const uint32_t packed_block_iter_base,
                                          uint32_t mma_kv_idx = 0) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  if constexpr (KTraits::NUM_MMA_KV == 1) {
    if (warpgroup_idx == 0) {
      bool row_mask;
      get_row_mask<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask);

      load_kv_r_partial<KTraits, Begin, End>(row_mask, kv_frag[0], kv, kv_offset[0]);
    }
  } else if constexpr (CTA_TILE_Q == 64) {
    if (warpgroup_idx == 0) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        bool row_mask;
        get_row_mask<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask,
                                          mma_kv);

        load_kv_r_partial<KTraits, Begin, End>(row_mask, kv_frag[mma_kv], kv, kv_offset[mma_kv]);
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      bool row_mask;
      get_row_mask<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask, mma_kv);

      load_kv_r_partial<KTraits, Begin, End>(row_mask, kv_frag[mma_kv], kv, kv_offset[mma_kv]);
    }
  }
}

// The purpose of this function is to load all kv.
template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void load_kv_r(typename KTraits::DTypeKV* ckv,
                                          typename KTraits::DTypeKV* kpe,
                                          uint32_t (*ckv_frag)[KTraits::NUM_MMA_D_CKV / 4][4],
                                          uint32_t (*kpe_frag)[KTraits::NUM_MMA_D_KPE / 4][4],
                                          int64_t* ckv_offset, int64_t* kpe_offset,
                                          const uint32_t packed_kv_bound,
                                          const uint32_t packed_block_iter_base) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  if constexpr (KTraits::NUM_MMA_KV == 1) {
    if (warpgroup_idx == 0) {
      bool row_mask;
      get_row_mask<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask);

      load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_CKV / 4>(row_mask, ckv_frag[0], ckv,
                                                                ckv_offset[0]);
      load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_KPE / 4>(row_mask, kpe_frag[0], kpe,
                                                                kpe_offset[0]);
    }
  } else if constexpr (CTA_TILE_Q == 64) {
    if (warpgroup_idx == 0) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        bool row_mask;
        get_row_mask<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask,
                                          mma_kv);

        load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_CKV / 4>(row_mask, ckv_frag[mma_kv], ckv,
                                                                  ckv_offset[mma_kv]);
        load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_KPE / 4>(row_mask, kpe_frag[mma_kv], kpe,
                                                                  kpe_offset[mma_kv]);
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      bool row_mask;
      get_row_mask<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask, mma_kv);

      load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_CKV / 4>(row_mask, ckv_frag[mma_kv], ckv,
                                                                ckv_offset[mma_kv]);
      load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_KPE / 4>(row_mask, kpe_frag[mma_kv], kpe,
                                                                kpe_offset[mma_kv]);
    }
  }
}

template <uint32_t NUM_MMA_D, uint32_t CTA_TILE_Q, SwizzleMode SWIZZLE_MODE_KV,
          uint32_t UPCAST_STRIDE_KV, bool LDS_TRANS_ENABLE = false>
__device__ __forceinline__ void load_kv_w_partial(uint32_t (*frag)[4],
                                                  smem_t<SWIZZLE_MODE_KV> kv_smem,
                                                  uint32_t mma_kv_idx = 0) {
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t kv_smem_offset_w = 0;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
    if constexpr (CTA_TILE_Q == 32) {
      static_assert(!LDS_TRANS_ENABLE, "When smem_size=128KB, we not support tile_q = 32");

      kv_smem_offset_w = kv_smem.template get_permuted_offset<UPCAST_STRIDE_KV>(
          32 * mma_kv_idx + warpgroup_idx * 16 + warp_idx_in_wg * 8 + lane_idx / 8,
          8 * mma_d + lane_idx % 8);
    } else {
      if constexpr (LDS_TRANS_ENABLE) {
        kv_smem_offset_w =
            kv_smem.template get_permuted_offset<UPCAST_STRIDE_KV, 4>(
                32 * mma_kv_idx + warpgroup_idx * 16 + warp_idx_in_wg * 8 + lane_idx / 8,
                (8 * mma_d + lane_idx % 8) / 2) +
            lane_idx % 2;
      } else {
        kv_smem_offset_w = kv_smem.template get_permuted_offset<UPCAST_STRIDE_KV>(
            warp_idx_in_wg * 8 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
      }
    }
    kv_smem.store_128b(kv_smem_offset_w, frag[mma_d]);
  }
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false>
__device__ __forceinline__ void load_kv_w(typename KTraits::SharedStorage* smem_storage,
                                          uint32_t (*ckv_frag)[KTraits::NUM_MMA_D_CKV / 4][4],
                                          uint32_t (*kpe_frag)[KTraits::NUM_MMA_D_KPE / 4][4],
                                          const uint32_t stage_idx) {
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  if constexpr (KTraits::NUM_MMA_KV == 1) {
    if (warpgroup_idx == 0) {
      load_kv_w_partial<NUM_MMA_D_CKV, CTA_TILE_Q, KTraits::SWIZZLE_MODE_CKV, UPCAST_STRIDE_CKV,
                        LDS_TRANS_ENABLE>(ckv_frag[0], ckv_smem);
      load_kv_w_partial<NUM_MMA_D_KPE, CTA_TILE_Q, KTraits::SWIZZLE_MODE_KPE, UPCAST_STRIDE_KPE,
                        LDS_TRANS_ENABLE>(kpe_frag[0], kpe_smem);
    }
  } else if constexpr (CTA_TILE_Q == 64) {
    if (warpgroup_idx == 0) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        load_kv_w_partial<NUM_MMA_D_CKV, CTA_TILE_Q, KTraits::SWIZZLE_MODE_CKV, UPCAST_STRIDE_CKV,
                          LDS_TRANS_ENABLE>(ckv_frag[mma_kv], ckv_smem);
        load_kv_w_partial<NUM_MMA_D_KPE, CTA_TILE_Q, KTraits::SWIZZLE_MODE_KPE, UPCAST_STRIDE_KPE,
                          LDS_TRANS_ENABLE>(kpe_frag[mma_kv], kpe_smem);
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      load_kv_w_partial<NUM_MMA_D_CKV, CTA_TILE_Q, KTraits::SWIZZLE_MODE_CKV, UPCAST_STRIDE_CKV,
                        LDS_TRANS_ENABLE>(ckv_frag[mma_kv], ckv_smem, mma_kv);
      load_kv_w_partial<NUM_MMA_D_KPE, CTA_TILE_Q, KTraits::SWIZZLE_MODE_KPE, UPCAST_STRIDE_KPE,
                        LDS_TRANS_ENABLE>(kpe_frag[mma_kv], kpe_smem, mma_kv);
    }
  }
}

template <typename KTraits, uint32_t NUM_MMA_D_QK, uint32_t UPCAST_STRIDE_Q,
          uint32_t UPCAST_STRIDE_K, SwizzleMode SWIZZLE_MODE_KV, bool LDS_TRANS_ENABLE = false,
          bool USE_LDGBSM = false>
__device__ __forceinline__ void compute_qk_(uint32_t (*q_frag)[NUM_MMA_D_QK / 2][4],
                                            smem_t<SWIZZLE_MODE_KV> k_smem,
                                            typename KTraits::DTypeQKAccum (*s_frag)[4]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  alignas(16) uint32_t k_frag[4];
  auto k_smem_r_swizzle = lane_idx / 16 % 2;
  if (lane_idx % 16 / 4 % 2 == 1) {
    k_smem_r_swizzle ^= 1;
  }
  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK / 2; ++mma_d) {
    if constexpr (KTraits::QK_SHARD) {
      uint32_t k_smem_offset_r;

#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        if constexpr (LDS_TRANS_ENABLE) {
          if constexpr (USE_LDGBSM) {
            k_smem_offset_r =
                k_smem.template get_swizzle_offset<true>(
                    mma_d / 2 * 64 +
                        (mma_kv * 4 + warpgroup_idx * 2 + lane_idx % 16 / 8) * UPCAST_STRIDE_K * 8,
                    lane_idx % 8, (mma_d % 2 * 4 + lane_idx / 16) / 2) +
                k_smem_r_swizzle;
          } else {
            k_smem_offset_r =
                k_smem.template get_permuted_offset<UPCAST_STRIDE_K, 4>(
                    (warpgroup_idx * (KTraits::NUM_MMA_KV / 2) + mma_kv) * 16 + lane_idx % 16,
                    (4 * mma_d + lane_idx / 16) / 2) +
                lane_idx / 16 % 2;
          }
        } else {
          k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
              (warpgroup_idx * (KTraits::NUM_MMA_KV / 2) + mma_kv) * 16 + lane_idx % 16,
              4 * mma_d + lane_idx / 16);
        }

        k_smem.load_128b(k_smem_offset_r, k_frag);

        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[mma_kv], q_frag[0][mma_d], k_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[mma_kv], q_frag[0][mma_d] + 2, k_frag + 2);
      }
    } else {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
        uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
            mma_kv * 16 + lane_idx % 16, 4 * mma_d + lane_idx / 16);

        k_smem.load_128b(k_smem_offset_r, k_frag);

        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[mma_kv], q_frag[0][mma_d], k_frag);
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[mma_kv], q_frag[0][mma_d] + 2, k_frag + 2);
      }
    }
  }
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false, bool USE_LDGBSM = false>
__device__ __forceinline__ void compute_mla_qk(
    typename KTraits::SharedStorage* smem_storage, const uint32_t stage_idx,
    uint32_t (*q_nope_frag)[KTraits::NUM_MMA_D_CKV / 2][4],
    uint32_t (*q_rope_frag)[KTraits::NUM_MMA_D_KPE / 2][4],
    typename KTraits::DTypeQKAccum (*s_frag)[4]) {
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  compute_qk_<KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE,
              KTraits::UPCAST_STRIDE_CKV, KTraits::SWIZZLE_MODE_KPE, LDS_TRANS_ENABLE, USE_LDGBSM>(
      q_nope_frag, ckv_smem, s_frag);
  compute_qk_<KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_Q_PE,
              KTraits::UPCAST_STRIDE_KPE, KTraits::SWIZZLE_MODE_CKV, LDS_TRANS_ENABLE, USE_LDGBSM>(
      q_rope_frag, kpe_smem, s_frag);
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false, bool USE_LDGBSM = false>
__device__ __forceinline__ void compute_mla_pv(
    typename KTraits::SharedStorage* smem_storage, const uint32_t stage_idx,
    typename KTraits::DTypeQKAccum (*s_frag)[4], typename KTraits::DTypeQKAccum* d,
    float (*o_frag)[4], uint32_t (*ckv_smem_offset_r)[KTraits::NUM_MMA_D_CKV / 2],
    uint32_t(*p_smem_offset_r)) {
  static_assert(LDS_TRANS_ENABLE && USE_LDGBSM,
                "This function only supports using ldstrans and ldgbsm.");

  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV_64B = KTraits::UPCAST_STRIDE_CKV_64B;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  if constexpr (KTraits::QK_SHARD) {
    smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
    constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P_64B;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t p_frag[2];
      p_smem.load_64b(p_smem_offset_r[mma_kv], p_frag);

      if constexpr (LDS_TRANS_ENABLE) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
          uint32_t v_frag[2];
          ckv_smem.load_64b_trans(ckv_smem_offset_r[mma_kv][mma_d], v_frag);
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d],
                                                                               p_frag, v_frag);
        }
      }
    }
  } else {
    // no need to store p_smem because all warpgroups are working on the same p
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV][4];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::template cast<4>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2 / 8; ++mma_d) {
        uint32_t v_frag[4][4];
#pragma unroll
        for (uint32_t r = 0; r < 4; r++) {
          uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
              mma_kv * 16 + lane_idx / 16 * 4 + r,
              lane_idx % 16 + mma_d * 16 + warpgroup_idx * UPCAST_STRIDE_CKV / 2);
          ckv_smem.load_128b(ckv_smem_offset_r, v_frag[r]);
        }

#pragma unroll
        for (uint32_t group = 0; group < 2; group++) {
          uint32_t perm_v[4][2];
          permute_128bx4(v_frag, perm_v, group);
#pragma unroll
          for (uint32_t i = 0; i < 4; i++) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
                o_frag[i + group * 4 + mma_d * 8], (uint32_t*)p_f16[mma_kv], perm_v[i]);
          }
        }
      }
    }
  }
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false, bool USE_LDGBSM = false>
__device__ __forceinline__ void compute_mla_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[4],
                                               typename KTraits::DTypeQKAccum* d,
                                               float (*o_frag)[4]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV_64B = KTraits::UPCAST_STRIDE_CKV_64B;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  if constexpr (KTraits::QK_SHARD) {
    smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
    constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P_64B;

#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      uint32_t p_frag[2];
      uint32_t p_smem_offset_r = p_smem.template get_permuted_offset_64b<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx % 16, mma_kv * 4 + lane_idx / 16);
      p_smem.load_64b(p_smem_offset_r, p_frag);

      if constexpr (LDS_TRANS_ENABLE) {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
          uint32_t v_frag[2];
          uint32_t ckv_smem_offset_r;
          if constexpr (USE_LDGBSM) {
            ckv_smem_offset_r = ckv_smem.template get_swizzle_offset_64b<true>(
                                    (mma_d / 4 + warpgroup_idx * NUM_MMA_D_CKV / 2 / 4) * 128 +
                                        (mma_kv * 2 + lane_idx / 32) * UPCAST_STRIDE_CKV_64B * 8,
                                    lane_idx / 4 % 8, mma_d % 4) +
                                lane_idx % 4;

            if (lane_idx / 16 % 2 == 1) {
              ckv_smem_offset_r ^= 2;
            }
          } else {
            ckv_smem_offset_r = ckv_smem.template get_permuted_offset_64b<UPCAST_STRIDE_CKV_64B, 4>(
                                    mma_kv * 16 + lane_idx / 4,
                                    mma_d + warpgroup_idx * UPCAST_STRIDE_CKV_64B / 2 / 4) +
                                lane_idx % 4;
          }

          ckv_smem.load_64b_trans(ckv_smem_offset_r, v_frag);
          mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[mma_d],
                                                                               p_frag, v_frag);
        }
      } else {
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2 / 8; ++mma_d) {
          uint32_t v_frag[4][4];
#pragma unroll
          for (uint32_t r = 0; r < 4; r++) {
            uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
                mma_kv * 16 + lane_idx / 16 * 4 + r,
                lane_idx % 16 + mma_d * 16 + warpgroup_idx * UPCAST_STRIDE_CKV / 2);
            ckv_smem.load_128b(ckv_smem_offset_r, v_frag[r]);
          }

#pragma unroll
          for (uint32_t group = 0; group < 2; group++) {
            uint32_t perm_v[4][2];
            permute_128bx4(v_frag, perm_v, group);
#pragma unroll
            for (uint32_t i = 0; i < 4; i++) {
              mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
                  o_frag[i + group * 4 + mma_d * 8], p_frag, perm_v[i]);
            }
          }
        }
      }
    }
  } else {
    // no need to store p_smem because all warpgroups are working on the same p
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV][4];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::template cast<4>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2 / 8; ++mma_d) {
        uint32_t v_frag[4][4];
#pragma unroll
        for (uint32_t r = 0; r < 4; r++) {
          uint32_t ckv_smem_offset_r = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
              mma_kv * 16 + lane_idx / 16 * 4 + r,
              lane_idx % 16 + mma_d * 16 + warpgroup_idx * UPCAST_STRIDE_CKV / 2);
          ckv_smem.load_128b(ckv_smem_offset_r, v_frag[r]);
        }

#pragma unroll
        for (uint32_t group = 0; group < 2; group++) {
          uint32_t perm_v[4][2];
          permute_128bx4(v_frag, perm_v, group);
#pragma unroll
          for (uint32_t i = 0; i < 4; i++) {
            mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(
                o_frag[i + group * 4 + mma_d * 8], (uint32_t*)p_f16[mma_kv], perm_v[i]);
          }
        }
      }
    }
  }
}

template <typename KTraits, SwizzleMode SWIZZLE_MODE_Q, uint32_t NUM_MMA_Q_PER_WAVE,
          uint32_t NUM_MMA_D, uint32_t UPCAST_STRIDE_Q>
__device__ __forceinline__ void load_q_smem_reg_(typename KTraits::DTypeQ* q_smem_ptr,
                                                 uint32_t (*q_frag)[NUM_MMA_D / 2][4]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  smem_t<SWIZZLE_MODE_Q> q_smem(q_smem_ptr);

  uint32_t q_smem_offset_r = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
      warp_idx_in_wg * 16 + lane_idx % 16, lane_idx / 16);

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 2; ++mma_d) {
#pragma unroll
    for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q_PER_WAVE; ++mma_q) {
      uint32_t* frag = &q_frag[mma_q][mma_d][0];
      q_smem.load_128b(q_smem_offset_r, frag);
      q_smem_offset_r = q_smem.template advance_offset_by_row<16, UPCAST_STRIDE_Q>(q_smem_offset_r);
    }
    q_smem_offset_r = q_smem.template advance_offset_by_column<4>(q_smem_offset_r, mma_d) -
                      NUM_MMA_Q_PER_WAVE * 16 * UPCAST_STRIDE_Q;
  }
}

template <typename KTraits, uint32_t NUM_MMA_D_CKV, uint32_t NUM_MMA_D_KPE>
__device__ __forceinline__ void load_q_smem_reg(typename KTraits::SharedStorage* smem_storage,
                                                uint32_t (*q_nope_frag)[NUM_MMA_D_CKV / 2][4],
                                                uint32_t (*q_rope_frag)[NUM_MMA_D_KPE / 2][4]) {
  constexpr uint32_t NUM_MMA_Q_PER_WAVE = KTraits::NUM_MMA_Q_PER_WAVE;
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;

  // lds q_nope
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_NOPE, NUM_MMA_Q_PER_WAVE, NUM_MMA_D_CKV,
                   UPCAST_STRIDE_Q_NOPE>(smem_storage->q_smem_nope, q_nope_frag);
  // lds q_rope
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_PE, NUM_MMA_Q_PER_WAVE, NUM_MMA_D_KPE,
                   UPCAST_STRIDE_Q_PE>(smem_storage->q_smem_pe, q_rope_frag);
}

template <typename KTraits, uint32_t NUM_MMA_D_CKV>
__device__ __forceinline__ void load_q_smem_reg_nope(
    typename KTraits::SharedStorage* smem_storage, uint32_t (*q_nope_frag)[NUM_MMA_D_CKV / 2][4]) {
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_NOPE, KTraits::NUM_MMA_Q_PER_WAVE,
                   NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE>(smem_storage->q_smem_nope,
                                                                 q_nope_frag);
}

template <typename KTraits, uint32_t NUM_MMA_D_KPE>
__device__ __forceinline__ void load_q_smem_reg_pe(typename KTraits::SharedStorage* smem_storage,
                                                   uint32_t (*q_rope_frag)[NUM_MMA_D_KPE / 2][4]) {
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_PE, KTraits::NUM_MMA_Q_PER_WAVE, NUM_MMA_D_KPE,
                   KTraits::UPCAST_STRIDE_Q_PE>(smem_storage->q_smem_pe, q_rope_frag);
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_FA2_UTILS_128B_CUH_