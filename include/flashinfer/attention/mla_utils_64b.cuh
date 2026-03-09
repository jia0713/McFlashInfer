/*
 * Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 */
#ifndef FLASHINFER_MLA_FA2_UTILS_64B_CUH_
#define FLASHINFER_MLA_FA2_UTILS_64B_CUH_

#include <cstdint>
#include <sstream>

#include "mla_params.cuh"
#include "prefill_utils.cuh"
#include "variant_helper.cuh"

namespace flashinfer {

namespace mla {

template <typename KTraits>
__device__ __forceinline__ void compute_kv_offset_64b(
    uint32_t* kv_page_idx, uint32_t* kv_page_offset, int64_t* ckv_offset, int64_t* kpe_offset,
    const uint64_t ckv_stride_n, const uint64_t ckv_stride_page, const uint64_t kpe_stride_n,
    const uint64_t kpe_stride_page) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  const uint32_t lane_idx = threadIdx.x;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV_PER_WAVE; ++mma_kv) {
    ckv_offset[mma_kv] = kv_page_idx[mma_kv] * ckv_stride_page +
                         kv_page_offset[mma_kv] * ckv_stride_n +
                         (lane_idx % 16) * upcast_size_64b<DTypeKV>();
    kpe_offset[mma_kv] = kv_page_idx[mma_kv] * kpe_stride_page +
                         kv_page_offset[mma_kv] * kpe_stride_n +
                         (lane_idx % 16) * upcast_size_64b<DTypeKV>();
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void prefetch_kv_indices_64b(
    const uint32_t packed_block_iter_base, const uint_fastdiv& block_size,
    const uint32_t packed_kv_bound, typename KTraits::IdType* indices, int64_t* ckv_offset,
    int64_t* kpe_offset, const uint64_t ckv_stride_n, const uint64_t ckv_stride_page,
    const uint64_t kpe_stride_n, const uint64_t kpe_stride_page) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_KV =
      (CTA_TILE_Q == 32) ? KTraits::NUM_MMA_KV : KTraits::NUM_MMA_KV / 2;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t kv_page_idx;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV_PER_WAVE; ++mma_kv) {
    uint32_t q, r;
    // uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 16 + 32 * mma_kv +
    //                              warpgroup_idx * 16 + warp_idx_in_wg * 4;
    uint32_t packed_block_iter =
        packed_block_iter_base + lane_idx / 16 + (CTA_TILE_Q == 32 ? 16 : 32) * mma_kv +
        (CTA_TILE_Q == 32 ? 2 : 4) * warpgroup_idx * 4 + warp_idx_in_wg * 4;
    block_size.divmod(packed_block_iter, q, r);
    bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
    cp_async::load_32b_pred(&kv_page_idx, indices + q, row_mask);
    ckv_offset[mma_kv] = kv_page_idx * ckv_stride_page + r * ckv_stride_n +
                         (lane_idx % 16) * upcast_size_64b<DTypeKV>();
    kpe_offset[mma_kv] = kv_page_idx * kpe_stride_page + r * kpe_stride_n +
                         (lane_idx % 16) * upcast_size_64b<DTypeKV>();
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void prefetch_kv_indices_64b(const uint32_t packed_block_iter_base,
                                                        const uint_fastdiv& block_size,
                                                        const uint32_t packed_kv_bound,
                                                        typename KTraits::IdType* indices,
                                                        uint32_t* kv_page_idx,
                                                        uint32_t* kv_page_offset) {
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_KV =
      (CTA_TILE_Q == 32) ? KTraits::NUM_MMA_KV : KTraits::NUM_MMA_KV / 2;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
    uint32_t q, r;
    // the general expression
    // uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 16 +
    //                              mma_kv * (num_warpgroups * num_warps_in_wg * 4) +
    //                              warpgroup_idx * num_warps_in_wg * 4 + warp_idx_in_wg * 4;
    uint32_t packed_block_iter =
        packed_block_iter_base + lane_idx / 16 + (CTA_TILE_Q == 32 ? 16 : 32) * mma_kv +
        (CTA_TILE_Q == 32 ? 2 : 4) * warpgroup_idx * 4 + warp_idx_in_wg * 4;
    block_size.divmod(packed_block_iter, q, r);
    bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
    cp_async::load_32b_pred(kv_page_idx + mma_kv, indices + q, row_mask);
    kv_page_offset[mma_kv] = r;
  }
}

template <typename KTraits, uint32_t Begin, uint32_t End>
__device__ __forceinline__ void load_kv_r_partial(bool row_mask, uint32_t (*frag)[2],
                                                  typename KTraits::DTypeKV* kv_ptr_base,
                                                  int64_t kv_offset) {
  using DTypeKV = typename KTraits::DTypeKV;
  const uint32_t lane_idx = threadIdx.x;
  DTypeKV* kv_ptr = kv_ptr_base + kv_offset;
  kv_ptr += Begin * 16 * upcast_size_64b<DTypeKV>();
#pragma unroll
  for (uint32_t mma_d = Begin; mma_d < End; ++mma_d) {
    cp_async::load_64b_pred(frag[mma_d], kv_ptr, row_mask);
    kv_ptr += 16 * upcast_size_64b<DTypeKV>();
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void get_row_mask_(const uint32_t packed_kv_bound,
                                              const uint32_t packed_block_iter_base, bool* row_mask,
                                              uint32_t mma_kv_idx = 0) {
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t packed_block_iter;

  packed_block_iter = packed_block_iter_base + lane_idx / 16 +
                      (CTA_TILE_Q == 32 ? 16 : 32) * mma_kv_idx +
                      (CTA_TILE_Q == 32 ? 2 : 4) * warpgroup_idx * 4 + warp_idx_in_wg * 4;

  *row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;
}

template <typename KTraits, uint32_t NUM_MMA_D, uint32_t Begin, uint32_t End,
          bool Is_even_MN = false>
__device__ __forceinline__ void load_kv_r(typename KTraits::DTypeKV* kv,
                                          uint32_t (*kv_frag)[NUM_MMA_D / 4][2], int64_t* kv_offset,
                                          const uint32_t packed_kv_bound,
                                          const uint32_t packed_block_iter_base,
                                          uint32_t mma_kv_idx = 0) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
    bool row_mask;
    get_row_mask_<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask, mma_kv);
    load_kv_r_partial<KTraits, Begin, End>(row_mask, kv_frag[mma_kv], kv, kv_offset[mma_kv]);
  }
}

template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void load_kv_r(typename KTraits::DTypeKV* ckv,
                                          typename KTraits::DTypeKV* kpe,
                                          uint32_t (*ckv_frag)[KTraits::NUM_MMA_D_CKV / 4][2],
                                          uint32_t (*kpe_frag)[KTraits::NUM_MMA_D_KPE / 4][2],
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

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
    bool row_mask;
    get_row_mask_<KTraits, Is_even_MN>(packed_kv_bound, packed_block_iter_base, &row_mask, mma_kv);

    load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_CKV / 4>(row_mask, ckv_frag[mma_kv], ckv,
                                                              ckv_offset[mma_kv]);
    load_kv_r_partial<KTraits, 0, KTraits::NUM_MMA_D_KPE / 4>(row_mask, kpe_frag[mma_kv], kpe,
                                                              kpe_offset[mma_kv]);
  }
}

template <uint32_t NUM_MMA_D, uint32_t CTA_TILE_Q, SwizzleMode SWIZZLE_MODE_KV,
          uint32_t UPCAST_STRIDE_KV>
__device__ __forceinline__ void load_kv_w_partial(uint32_t (*frag)[2],
                                                  smem_t<SWIZZLE_MODE_KV> kv_smem,
                                                  uint32_t mma_kv_idx) {
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t kv_smem_offset_w = 0;
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
    kv_smem_offset_w = kv_smem.template get_permuted_offset_64b<UPCAST_STRIDE_KV>(
        (CTA_TILE_Q == 32 ? 16 : 32) * mma_kv_idx + (CTA_TILE_Q == 32 ? 2 : 4) * warpgroup_idx * 4 +
            warp_idx_in_wg * 4 + lane_idx / 16,
        16 * mma_d + lane_idx % 16);
    kv_smem.store_64b(kv_smem_offset_w, frag[mma_d]);
  }

  // uint32_t kv_smem_offset_w = kv_smem.template get_permuted_offset_64b<UPCAST_STRIDE_KV>(
  // warpgroup_idx * 16 + warp_idx_in_wg * 4 + lane_idx / 16, lane_idx % 16);

  // #pragma unroll
  //   for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
  //     kv_smem.store_64b(kv_smem_offset_w, frag[mma_d]);
  //     kv_smem_offset_w = kv_smem.template advance_offset_by_column<16>(kv_smem_offset_w);
  //   }
}

template <typename KTraits>
__device__ __forceinline__ void load_kv_w(typename KTraits::SharedStorage* smem_storage,
                                          uint32_t (*ckv_frag)[KTraits::NUM_MMA_D_CKV / 4][2],
                                          uint32_t (*kpe_frag)[KTraits::NUM_MMA_D_KPE / 4][2],
                                          const uint32_t stage_idx) {
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV_64B;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE_64B;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  constexpr uint32_t NUM_MMA_KV = CTA_TILE_Q == 32 ? KTraits::NUM_MMA_KV : KTraits::NUM_MMA_KV / 2;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  __builtin_mxc_schedbound_begin();
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
    load_kv_w_partial<NUM_MMA_D_CKV, CTA_TILE_Q, KTraits::SWIZZLE_MODE_CKV, UPCAST_STRIDE_CKV>(
        ckv_frag[mma_kv], ckv_smem, mma_kv);
    load_kv_w_partial<NUM_MMA_D_KPE, CTA_TILE_Q, KTraits::SWIZZLE_MODE_KPE, UPCAST_STRIDE_KPE>(
        kpe_frag[mma_kv], kpe_smem, mma_kv);
  }
  __builtin_mxc_schedbound_end();
}

template <typename KTraits>
__device__ __forceinline__ void get_k_base_offset_r(typename KTraits::SharedStorage* smem_storage,
                                                    uint32_t ckv_offset[], uint32_t kpe_offset[]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[0]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[0]);

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < 4; ++mma_d) {
    ckv_offset[mma_d] = ckv_smem.template get_permuted_offset_64b<KTraits::UPCAST_STRIDE_CKV_64B>(
        warpgroup_idx * 16 + lane_idx % 16, 4 * mma_d + lane_idx / 16);
    kpe_offset[mma_d] = kpe_smem.template get_permuted_offset_64b<KTraits::UPCAST_STRIDE_KPE_64B>(
        warpgroup_idx * 16 + lane_idx % 16, 4 * mma_d + lane_idx / 16);
  }
}

template <typename KTraits, uint32_t NUM_MMA_D_QK, uint32_t UPCAST_STRIDE_K,
          SwizzleMode SWIZZLE_MODE_KV>
__device__ __forceinline__ void compute_qk_(uint32_t (*q_frag)[NUM_MMA_D_QK][2],
                                            smem_t<SWIZZLE_MODE_KV> k_smem,
                                            typename KTraits::DTypeQKAccum (*s_frag)[4],
                                            const uint32_t k_offset[]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  alignas(16) uint32_t k_frag[2];

  static_assert(KTraits::NUM_MMA_KV == 2);
  static_assert(KTraits::QK_SHARD == true);
  uint32_t k_smem_offset_r[4] = {k_offset[0], k_offset[1], k_offset[2], k_offset[3]};

  // compute q*k^T
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_QK / 4; ++mma_d) {
#pragma unroll
    for (uint32_t d = 0; d < 4; ++d) {
      k_smem.load_64b(k_smem_offset_r[d], k_frag);
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
          s_frag[0], q_frag[0][mma_d * 4 + d], k_frag);
      k_smem_offset_r[d] = k_smem.template advance_offset_by_column<16>(k_smem_offset_r[d]);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_qk(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               uint32_t (*q_nope_frag)[KTraits::NUM_MMA_D_CKV][2],
                                               uint32_t (*q_rope_frag)[KTraits::NUM_MMA_D_KPE][2],
                                               typename KTraits::DTypeQKAccum (*s_frag)[4],
                                               const uint32_t ckv_offset[],
                                               const uint32_t kpe_offset[]) {
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  compute_qk_<KTraits, KTraits::NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_CKV_64B,
              KTraits::SWIZZLE_MODE_KPE>(q_nope_frag, ckv_smem, s_frag, ckv_offset);
  compute_qk_<KTraits, KTraits::NUM_MMA_D_KPE, KTraits::UPCAST_STRIDE_KPE_64B,
              KTraits::SWIZZLE_MODE_CKV>(q_rope_frag, kpe_smem, s_frag, kpe_offset);
}

template <typename KTraits>
__device__ __forceinline__ void get_v_base_offset_r(typename KTraits::SharedStorage* smem_storage,
                                                    uint32_t v_offset[]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[0]);

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < 4; ++mma_d) {
    v_offset[mma_d] = ckv_smem.template get_permuted_offset_64b<KTraits::UPCAST_STRIDE_CKV_64B>(
        lane_idx / 16 * 4 + mma_d,
        lane_idx % 16 + warpgroup_idx * KTraits::UPCAST_STRIDE_CKV_64B / 2);
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_mla_pv(typename KTraits::SharedStorage* smem_storage,
                                               const uint32_t stage_idx,
                                               typename KTraits::DTypeQKAccum (*s_frag)[4],
                                               typename KTraits::DTypeQKAccum* d,
                                               float (*o_frag)[4], const uint32_t v_offset[]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV_64B = KTraits::UPCAST_STRIDE_CKV_64B;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  static_assert(KTraits::QK_SHARD == true);

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
  constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P_64B;
  uint32_t ckv_smem_offset_r[4] = {v_offset[0], v_offset[1], v_offset[2], v_offset[3]};

#pragma unroll
  for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
    uint32_t p_frag[2];
    uint32_t p_smem_offset_r = p_smem.template get_permuted_offset_64b<UPCAST_STRIDE_P>(
        warp_idx_in_wg * 16 + lane_idx % 16, mma_kv * 4 + lane_idx / 16);
    p_smem.load_64b(p_smem_offset_r, p_frag);

#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2 / 4; ++mma_d) {
      uint32_t v_frag[4][2];
#pragma unroll
      for (uint32_t r = 0; r < 4; r++) {
        ckv_smem.load_64b(ckv_smem_offset_r[r], v_frag[r]);
        ckv_smem_offset_r[r] = ckv_smem.template advance_offset_by_column<16>(ckv_smem_offset_r[r]);
      }

      uint32_t perm_v[4][2];
      permute_64bx4(v_frag, perm_v);
#pragma unroll
      for (uint32_t i = 0; i < 4; i++) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeKV>(o_frag[i + mma_d * 4],
                                                                             p_frag, perm_v[i]);
      }
    }
#pragma unroll
    for (uint32_t r = 0; r < 4; r++) {
      ckv_smem_offset_r[r] = ckv_smem_offset_r[r] - NUM_MMA_D_CKV * 2 + 16 * UPCAST_STRIDE_CKV_64B;
    }
  }
}

template <typename KTraits, SwizzleMode SWIZZLE_MODE_Q, uint32_t NUM_MMA_Q_PER_WAVE,
          uint32_t NUM_MMA_D, uint32_t UPCAST_STRIDE_Q>
__device__ __forceinline__ void load_q_smem_reg_(typename KTraits::DTypeQ* q_smem_ptr,
                                                 uint32_t (*q_frag)[NUM_MMA_D][2]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  smem_t<SWIZZLE_MODE_Q> q_smem(q_smem_ptr);

  static_assert(NUM_MMA_Q_PER_WAVE == 1, "NUM_MMA_Q_PER_WAVE must be 1");

#pragma unroll
  for (uint32_t d = 0; d < 4; ++d) {
    uint32_t q_smem_offset_r = q_smem.template get_permuted_offset_64b<UPCAST_STRIDE_Q, 8>(
                                   warp_idx_in_wg * 16 + lane_idx % 16, lane_idx / 32 + 2 * d) +
                               lane_idx / 16 % 2;
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
      q_smem.load_64b(q_smem_offset_r, &q_frag[0][mma_d * 4 + d][0]);
      q_smem_offset_r = q_smem.template advance_offset_by_column<16>(q_smem_offset_r, mma_d);
    }
  }
}

template <typename KTraits, uint32_t NUM_MMA_D_CKV, uint32_t NUM_MMA_D_KPE>
__device__ __forceinline__ void load_q_smem_reg(typename KTraits::SharedStorage* smem_storage,
                                                uint32_t (*q_nope_frag)[NUM_MMA_D_CKV][2],
                                                uint32_t (*q_rope_frag)[NUM_MMA_D_KPE][2]) {
  constexpr uint32_t NUM_MMA_Q_PER_WAVE = KTraits::NUM_MMA_Q_PER_WAVE;
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE_64B;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE_64B;

  // lds q_nope
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_NOPE, NUM_MMA_Q_PER_WAVE, NUM_MMA_D_CKV,
                   UPCAST_STRIDE_Q_NOPE>(smem_storage->q_smem_nope, q_nope_frag);
  // lds q_rope
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_PE, NUM_MMA_Q_PER_WAVE, NUM_MMA_D_KPE,
                   UPCAST_STRIDE_Q_PE>(smem_storage->q_smem_pe, q_rope_frag);
}

template <typename KTraits, uint32_t NUM_MMA_D_CKV>
__device__ __forceinline__ void load_q_smem_reg_nope(typename KTraits::SharedStorage* smem_storage,
                                                     uint32_t (*q_nope_frag)[NUM_MMA_D_CKV][2]) {
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_NOPE, KTraits::NUM_MMA_Q_PER_WAVE,
                   NUM_MMA_D_CKV, KTraits::UPCAST_STRIDE_Q_NOPE_64B>(smem_storage->q_smem_nope,
                                                                     q_nope_frag);
}

template <typename KTraits, uint32_t NUM_MMA_D_KPE>
__device__ __forceinline__ void load_q_smem_reg_pe(typename KTraits::SharedStorage* smem_storage,
                                                   uint32_t (*q_rope_frag)[NUM_MMA_D_KPE][2]) {
  load_q_smem_reg_<KTraits, KTraits::SWIZZLE_MODE_Q_PE, KTraits::NUM_MMA_Q_PER_WAVE, NUM_MMA_D_KPE,
                   KTraits::UPCAST_STRIDE_Q_PE_64B>(smem_storage->q_smem_pe, q_rope_frag);
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_FA2_UTILS_64B_CUH_