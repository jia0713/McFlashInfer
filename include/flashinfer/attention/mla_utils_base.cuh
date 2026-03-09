/*
 * Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 */
#ifndef FLASHINFER_MLA_FA2_UTILS_BASE_CUH_
#define FLASHINFER_MLA_FA2_UTILS_BASE_CUH_

#include "mla_utils_128b.cuh"
#include "mla_utils_64b.cuh"

namespace flashinfer {

namespace mla {

struct StandardAttention : AttentionVariantBase {
  float sm_scale_log2;

  template <typename Params>
  __device__ __host__ StandardAttention(const Params& params, uint32_t batch_idx,
                                        uint8_t* smem_ptr) {
    sm_scale_log2 = params.sm_scale * math::log2e;
  }
};

template <uint32_t NUM_STAGES, uint32_t CTA_TILE_Q, uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV,
          uint32_t HEAD_DIM_KPE, typename DTypeQ, typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO {
  union {
    struct {
      alignas(16) DTypeQ q_smem_nope[CTA_TILE_Q * HEAD_DIM_CKV];
      alignas(16) DTypeQ q_smem_pe[CTA_TILE_Q * HEAD_DIM_KPE];
    };
    struct {
      alignas(16) DTypeKV ckv_smem[NUM_STAGES][CTA_TILE_KV * HEAD_DIM_CKV];
      alignas(16) DTypeKV
          kpe_p_smem[NUM_STAGES]
                    [CTA_TILE_KV * (HEAD_DIM_KPE > CTA_TILE_Q ? HEAD_DIM_KPE : CTA_TILE_Q)];
    };
    alignas(16) DTypeO o_smem[CTA_TILE_Q * HEAD_DIM_CKV];
  };
  union {
    alignas(16) float m_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
    alignas(16) float d_wg[2][CTA_TILE_Q];  // cross warpgroup synchronization
  };
};

template <uint32_t CTA_TILE_KV, uint32_t HEAD_DIM_CKV, uint32_t HEAD_DIM_KPE, typename DTypeQ,
          typename DTypeKV, typename DTypeO>
struct SharedStorageQKVO<1, 64, CTA_TILE_KV, HEAD_DIM_CKV, HEAD_DIM_KPE, DTypeQ, DTypeKV, DTypeO> {
  union {
    struct {
      alignas(16) DTypeKV ckv_smem[1][CTA_TILE_KV * HEAD_DIM_CKV];
      alignas(16) DTypeKV kpe_p_smem[1][CTA_TILE_KV * (HEAD_DIM_KPE > 64 ? HEAD_DIM_KPE : 64)];
      alignas(16) float m_wg[2][64];  // cross warpgroup synchronization
      alignas(16) float d_wg[2][64];  // cross warpgroup synchronization
      alignas(16) DTypeQ q_smem_pe[64 * HEAD_DIM_KPE];
    };
    alignas(16) DTypeQ q_smem_nope[64 * HEAD_DIM_CKV];
    alignas(16) DTypeO o_smem[64 * HEAD_DIM_CKV];
  };
};

template <bool CAUSAL_, uint32_t NUM_STAGES_, bool QK_SHARD_, uint32_t HEAD_DIM_CKV_,
          uint32_t HEAD_DIM_KPE_, uint32_t CTA_TILE_Q_, uint32_t CTA_TILE_KV_, typename DTypeQ_,
          typename DTypeKV_, typename DTypeO_, typename IdType_>
struct KernelTraits {
  static constexpr bool CAUSAL = CAUSAL_;
  static constexpr uint32_t NUM_STAGES = NUM_STAGES_;
  // NOTE(Zihao): whether to shard Q*K computation across warpgroups
  // if true, each warpgroup will compute a subset of Q*K (sharded on the KV dimension)
  // if false, each warpgroup will compute the full Q*K, which is duplicated across warpgroups
  // when CTA_TILE_KV / 16 <= warpgroup nums, QK_SHARD only support false
  static constexpr bool QK_SHARD = QK_SHARD_;
  static constexpr uint32_t NUM_MMA_Q = CTA_TILE_Q_ / 16;
  static constexpr uint32_t NUM_MMA_Q_PER_WAVE = 1;
  static constexpr uint32_t NUM_MMA_KV = CTA_TILE_KV_ / 16;
  static constexpr uint32_t NUM_MMA_KV_PER_WAVE = QK_SHARD ? NUM_MMA_KV / 2 : NUM_MMA_KV;
  static constexpr uint32_t HEAD_DIM_CKV = HEAD_DIM_CKV_;
  static constexpr uint32_t HEAD_DIM_KPE = HEAD_DIM_KPE_;
  static constexpr uint32_t HEAD_DIM_ALL = HEAD_DIM_CKV + HEAD_DIM_KPE;
  static constexpr uint32_t NUM_MMA_D_CKV = HEAD_DIM_CKV / 16;
  static constexpr uint32_t NUM_MMA_D_KPE = HEAD_DIM_KPE / 16;
  static constexpr uint32_t NUM_THREADS = CTA_TILE_Q_ == 64 ? 512 : 256;
  static constexpr uint32_t CTA_TILE_Q = CTA_TILE_Q_;
  static constexpr uint32_t CTA_TILE_KV = CTA_TILE_KV_;

  static constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_Q_PE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_CKV = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_KPE = SwizzleMode::k128B;
  static constexpr SwizzleMode SWIZZLE_MODE_P =
      CTA_TILE_KV >= 64 ? SwizzleMode::k128B : SwizzleMode::k64B;
  static constexpr SwizzleMode SWIZZLE_MODE_O = SwizzleMode::k128B;
  static constexpr uint32_t UPCAST_STRIDE_Q_NOPE = HEAD_DIM_CKV / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_Q_PE = HEAD_DIM_KPE / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_CKV = HEAD_DIM_CKV / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_CKV_64B = HEAD_DIM_CKV / upcast_size_64b<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_KPE = HEAD_DIM_KPE / upcast_size<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_KPE_64B = HEAD_DIM_KPE / upcast_size_64b<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_FINAL_O = HEAD_DIM_CKV / upcast_size<DTypeO_>();
  static constexpr uint32_t UPCAST_STRIDE_FINAL_O_64B = HEAD_DIM_CKV / upcast_size_64b<DTypeO_>();
  static constexpr uint32_t UPCAST_STRIDE_P_64B = CTA_TILE_KV / upcast_size_64b<DTypeKV_>();
  static constexpr uint32_t UPCAST_STRIDE_PARTIAL_O = HEAD_DIM_CKV / upcast_size<float>();

  static constexpr uint32_t UPCAST_STRIDE_Q_NOPE_64B = HEAD_DIM_CKV / upcast_size_64b<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_Q_PE_64B = HEAD_DIM_KPE / upcast_size_64b<DTypeQ_>();

  using DTypeQ = DTypeQ_;
  using DTypeKV = DTypeKV_;
  using DTypeO = DTypeO_;
  using IdType = IdType_;
  using DTypeQKAccum = float;

  using SharedStorage = SharedStorageQKVO<NUM_STAGES, CTA_TILE_Q, CTA_TILE_KV, HEAD_DIM_CKV,
                                          HEAD_DIM_KPE, DTypeQ, DTypeKV, DTypeO>;
  using AttentionVariant = StandardAttention;

  static constexpr DTypeQKAccum MaskFillValue = -math::inf;
};

template <typename KTraits>
__device__ __forceinline__ void init_states_(float (*o_frag)[4], typename KTraits::DTypeQKAccum* m,
                                             float* d) {
#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
#pragma unroll
    for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
      o_frag[mma_d][reg_id] = 0.f;
    }
  }
  m[0] = typename KTraits::DTypeQKAccum(-math::inf);
  d[0] = 1.f;
}

template <typename KTraits>
__device__ __forceinline__ void load_q(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeQ* q_nope,
    typename KTraits::DTypeQ* q_pe, const uint32_t q_nope_stride_n, const uint32_t q_nope_stride_h,
    const uint32_t q_pe_stride_n, const uint32_t q_pe_stride_h, const uint32_t q_len,
    const uint32_t packed_offset, const uint_fastdiv& num_heads) {
  constexpr uint32_t UPCAST_STRIDE_Q_NOPE = KTraits::UPCAST_STRIDE_Q_NOPE;
  constexpr uint32_t UPCAST_STRIDE_Q_PE = KTraits::UPCAST_STRIDE_Q_PE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;

  load_q_partial<KTraits, UPCAST_STRIDE_Q_NOPE, NUM_MMA_D_CKV>(
      smem_storage, q_nope, q_nope_stride_n, q_nope_stride_h, q_len, packed_offset, num_heads);

  load_q_partial<KTraits, UPCAST_STRIDE_Q_PE, NUM_MMA_D_KPE>(
      smem_storage, q_pe, q_pe_stride_n, q_pe_stride_h, q_len, packed_offset, num_heads);
}

template <typename KTraits, uint32_t UPCAST_STRIDE_Q, uint32_t NUM_MMA_D>
__device__ __forceinline__ void load_q_partial(typename KTraits::SharedStorage* smem_storage,
                                               typename KTraits::DTypeQ* q_gmem,
                                               const uint32_t q_stride_n, const uint32_t q_stride_h,
                                               const uint32_t q_len, const uint32_t packed_offset,
                                               const uint_fastdiv& num_heads) {
  using DTypeQ = typename KTraits::DTypeQ;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  constexpr uint32_t WARPGROUP_SIZE = KTraits::CTA_TILE_Q / 16;
  // Only when swizzle==k128, Q_THR_LAYOUT_ROW=8. When modify Swizzle, you need to modify
  // Q_THR_LAYOUT_ROW.
  constexpr uint32_t Q_THR_LAYOUT_ROW = 8;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;

  DTypeQ* q_smem_ptr =
      (NUM_MMA_D == KTraits::NUM_MMA_D_CKV) ? smem_storage->q_smem_nope : smem_storage->q_smem_pe;

  smem_t<SwizzleMode::k128B> q_smem(q_smem_ptr);
  uint32_t q_frag[4];  // 8 half

#pragma unroll
  for (uint32_t mma_q = 0; mma_q < 1; ++mma_q) {
    uint32_t q, r;
    num_heads.divmod(packed_offset + lane_idx / 8 + CTA_TILE_Q * mma_q +
                         warpgroup_idx * Q_THR_LAYOUT_ROW * WARPGROUP_SIZE + warp_idx_in_wg * 8,
                     q, r);
    DTypeQ* q_ptr =
        q_gmem + q * q_stride_n + r * q_stride_h + (lane_idx % 8) * upcast_size<DTypeQ>();

#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D / 4; ++mma_d) {
      uint32_t q_smem_offset_w = q_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
          CTA_TILE_Q * mma_q + warpgroup_idx * Q_THR_LAYOUT_ROW * WARPGROUP_SIZE +
              warp_idx_in_wg * 8 + lane_idx / 8,
          mma_d * 8 + lane_idx % 8);
      cp_async::load_128b_pred(q_frag, q_ptr, q < q_len);
      q_smem.store_128b(q_smem_offset_w, q_frag);
      q_ptr += 8 * upcast_size<DTypeQ>();
    }
  }
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false, bool USE_LDGBSM = false>
__device__ __forceinline__ void get_kv_offset(
    typename KTraits::SharedStorage* smem_storage,
    uint32_t (*kv_gmem_offset_r)[KTraits::NUM_MMA_D_CKV / 4],
    uint32_t (*ckv_smem_offset_w)[KTraits::NUM_MMA_D_CKV / 4],
    uint32_t (*kpe_smem_offset_w)[KTraits::NUM_MMA_D_KPE / 4],
    uint32_t (*ckv_smem_offset_r)[KTraits::NUM_MMA_D_CKV / 2], uint32_t(*p_smem_offset_r)) {
  static_assert(USE_LDGBSM, "Only support ldgbsm.");
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t UPCAST_STRIDE_CKV_64B = KTraits::UPCAST_STRIDE_CKV_64B;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[0]);
  if (warpgroup_idx == 0) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        kv_gmem_offset_r[mma_kv][mma_d] =
            cp_async::get_permuted_offset<4>(lane_idx / 8, mma_d * 4 + lane_idx % 8 / 2) +
            lane_idx % 2;

        if constexpr (LDS_TRANS_ENABLE) {
          if (lane_idx / 32) {
            kv_gmem_offset_r[mma_kv][mma_d] ^= 1;
          }
        }

        kv_gmem_offset_r[mma_kv][mma_d] *= upcast_size<DTypeKV>();
        ckv_smem_offset_w[mma_kv][mma_d] =
            UPCAST_STRIDE_CKV * (warp_idx_in_wg * 8 + mma_kv * 32) + mma_d * 64 + lane_idx;
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        kpe_smem_offset_w[mma_kv][mma_d] =
            UPCAST_STRIDE_KPE * (warp_idx_in_wg * 8 + mma_kv * 32) + mma_d * 64 + lane_idx;
      }
    }
  }

  if constexpr (KTraits::QK_SHARD) {
    smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[0]);
    constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P_64B;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      p_smem_offset_r[mma_kv] = p_smem.template get_permuted_offset_64b<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx % 16, mma_kv * 4 + lane_idx / 16);
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
        ckv_smem_offset_r[mma_kv][mma_d] =
            ckv_smem.template get_swizzle_offset_64b<true>(
                (mma_d / 4 + warpgroup_idx * NUM_MMA_D_CKV / 2 / 4) * 128 +
                    (mma_kv * 2 + lane_idx / 32) * UPCAST_STRIDE_CKV_64B * 8,
                lane_idx / 4 % 8, mma_d % 4) +
            lane_idx % 4;
        if (lane_idx / 16 % 2 == 1) {
          ckv_smem_offset_r[mma_kv][mma_d] ^= 2;
        }
      }
    }
  }
}

// This function only supports using ldstrans and ldgbsm.
template <typename KTraits, bool Is_even_MN = false>
__device__ __forceinline__ void load_kv(typename KTraits::SharedStorage* smem_storage,
                                        typename KTraits::DTypeKV**(ckv_base_ptr),
                                        typename KTraits::DTypeKV*(*kpe_base_ptr),
                                        const uint32_t packed_kv_bound,
                                        const uint32_t packed_block_iter_base,
                                        const uint32_t stage_idx,
                                        uint32_t (*kv_gmem_offset_r)[KTraits::NUM_MMA_D_CKV / 4],
                                        uint32_t (*ckv_smem_offset_w)[KTraits::NUM_MMA_D_CKV / 4],
                                        uint32_t (*kpe_smem_offset_w)[KTraits::NUM_MMA_D_KPE / 4]) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t packed_block_iter;

  if constexpr (!Is_even_MN) {
    packed_block_iter =
        packed_block_iter_base + lane_idx / 8 + warpgroup_idx * 32 + warp_idx_in_wg * 8;
  }

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  if (warpgroup_idx == 0) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        if constexpr (Is_even_MN) {
          ckv_smem.template load_128b_async<typename KTraits::DTypeKV, Is_even_MN>(
              ckv_smem_offset_w[mma_kv][mma_d],
              ckv_base_ptr[mma_kv] + kv_gmem_offset_r[mma_kv][mma_d]);
        } else {
          ckv_smem.template load_128b_async<typename KTraits::DTypeKV, Is_even_MN>(
              ckv_smem_offset_w[mma_kv][mma_d],
              ckv_base_ptr[mma_kv] + kv_gmem_offset_r[mma_kv][mma_d],
              packed_block_iter < packed_kv_bound);
        }
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        if constexpr (Is_even_MN) {
          kpe_smem.template load_128b_async<typename KTraits::DTypeKV, Is_even_MN>(
              kpe_smem_offset_w[mma_kv][mma_d],
              kpe_base_ptr[mma_kv] + kv_gmem_offset_r[mma_kv][mma_d]);
        } else {
          kpe_smem.template load_128b_async<typename KTraits::DTypeKV, Is_even_MN>(
              kpe_smem_offset_w[mma_kv][mma_d],
              kpe_base_ptr[mma_kv] + kv_gmem_offset_r[mma_kv][mma_d],
              packed_block_iter < packed_kv_bound);
        }
      }

      if constexpr (!Is_even_MN) {
        packed_block_iter += 64;
      }
    }
  }
}

template <typename KTraits, bool Is_even_MN = false, bool LDS_TRANS_ENABLE = false>
__device__ __forceinline__ void load_kv(
    typename KTraits::SharedStorage* smem_storage, typename KTraits::DTypeKV* ckv,
    typename KTraits::DTypeKV* kpe, const uint32_t ckv_stride_n, const uint32_t ckv_stride_page,
    const uint32_t kpe_stride_n, const uint32_t kpe_stride_page, const uint32_t packed_kv_bound,
    const uint32_t packed_block_iter_base, const uint32_t stage_idx, uint32_t* kv_page_idx,
    uint32_t* kv_page_offset) {
  using DTypeKV = typename KTraits::DTypeKV;
  constexpr uint32_t UPCAST_STRIDE_CKV = KTraits::UPCAST_STRIDE_CKV;
  constexpr uint32_t UPCAST_STRIDE_KPE = KTraits::UPCAST_STRIDE_KPE;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  const uint32_t lane_idx = threadIdx.x;
  const uint32_t warpgroup_idx = threadIdx.z;
  const uint32_t warp_idx_in_wg = threadIdx.y;
  uint32_t k_frag[4];

  smem_t<KTraits::SWIZZLE_MODE_CKV> ckv_smem(smem_storage->ckv_smem[stage_idx]);
  smem_t<KTraits::SWIZZLE_MODE_KPE> kpe_smem(smem_storage->kpe_p_smem[stage_idx]);
  if constexpr (KTraits::NUM_MMA_KV == 1) {
    if (warpgroup_idx == 0) {
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + warp_idx_in_wg * 8;
      bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;

      DTypeKV* ckv_ptr = ckv + kv_page_idx[0] * ckv_stride_page + kv_page_offset[0] * ckv_stride_n +
                         (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr = kpe + kv_page_idx[0] * kpe_stride_page + kv_page_offset[0] * kpe_stride_n +
                         (lane_idx % 8) * upcast_size<DTypeKV>();

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            warp_idx_in_wg * 8 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
        cp_async::load_128b_pred(k_frag, ckv_ptr, row_mask);
        ckv_smem.store_128b(ckv_smem_offset_w, k_frag);
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
            warp_idx_in_wg * 8 + lane_idx / 8, 8 * mma_d + lane_idx % 8);
        cp_async::load_128b_pred(k_frag, kpe_ptr, row_mask);
        kpe_smem.store_128b(kpe_smem_offset_w, k_frag);
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  } else if constexpr (CTA_TILE_Q == 64) {
    if (warpgroup_idx == 0) {
#pragma unroll
      for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
        uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + 64 * mma_kv +
                                     warpgroup_idx * 32 + warp_idx_in_wg * 8;
        bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;

        DTypeKV* ckv_ptr_base =
            ckv + kv_page_idx[mma_kv] * ckv_stride_page + kv_page_offset[mma_kv] * ckv_stride_n;
        DTypeKV* kpe_ptr_base =
            kpe + kv_page_idx[mma_kv] * kpe_stride_page + kv_page_offset[mma_kv] * kpe_stride_n;

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
          uint32_t ckv_offset_r =
              cp_async::get_permuted_offset<4>(lane_idx / 8, mma_d * 4 + lane_idx % 8 / 2) +
              lane_idx % 2;

          if constexpr (LDS_TRANS_ENABLE) {
            if (lane_idx / 32) {
              ckv_offset_r ^= 1;
            }
          }

          uint32_t ckv_smem_offset_w =
              UPCAST_STRIDE_CKV * (warp_idx_in_wg * 8 + mma_kv * 32) + mma_d * 64 + lane_idx;
          ckv_smem.load_128b_async(ckv_smem_offset_w,
                                   ckv_ptr_base + ckv_offset_r * upcast_size<DTypeKV>(), row_mask);
        }

#pragma unroll
        for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
          uint32_t kpe_offset_r =
              cp_async::get_permuted_offset<4>(lane_idx / 8, mma_d * 4 + lane_idx % 8 / 2) +
              lane_idx % 2;
          uint32_t kpe_smem_offset_w =
              UPCAST_STRIDE_KPE * (warp_idx_in_wg * 8 + mma_kv * 32) + mma_d * 64 + lane_idx;
          kpe_smem.load_128b_async(kpe_smem_offset_w,
                                   kpe_ptr_base + kpe_offset_r * upcast_size<DTypeKV>(), row_mask);
        }
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      uint32_t packed_block_iter = packed_block_iter_base + lane_idx / 8 + 32 * mma_kv +
                                   warpgroup_idx * 16 + warp_idx_in_wg * 8;
      bool row_mask = Is_even_MN || packed_block_iter < packed_kv_bound;

      DTypeKV* ckv_ptr = ckv + kv_page_idx[mma_kv] * ckv_stride_page +
                         kv_page_offset[mma_kv] * ckv_stride_n +
                         (lane_idx % 8) * upcast_size<DTypeKV>();
      DTypeKV* kpe_ptr = kpe + kv_page_idx[mma_kv] * kpe_stride_page +
                         kv_page_offset[mma_kv] * kpe_stride_n +
                         (lane_idx % 8) * upcast_size<DTypeKV>();

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 4; ++mma_d) {
        uint32_t ckv_smem_offset_w = ckv_smem.template get_permuted_offset<UPCAST_STRIDE_CKV>(
            32 * mma_kv + warpgroup_idx * 16 + warp_idx_in_wg * 8 + lane_idx / 8,
            8 * mma_d + lane_idx % 8);
        cp_async::load_128b_pred(k_frag, ckv_ptr, row_mask);
        ckv_smem.store_128b(ckv_smem_offset_w, k_frag);
        ckv_ptr += 8 * upcast_size<DTypeKV>();
      }

#pragma unroll
      for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_KPE / 4; ++mma_d) {
        uint32_t kpe_smem_offset_w = kpe_smem.template get_permuted_offset<UPCAST_STRIDE_KPE>(
            32 * mma_kv + warpgroup_idx * 16 + warp_idx_in_wg * 8 + lane_idx / 8,
            8 * mma_d + lane_idx % 8);
        cp_async::load_128b_pred(k_frag, kpe_ptr, row_mask);
        kpe_smem.store_128b(kpe_smem_offset_w, k_frag);
        kpe_ptr += 8 * upcast_size<DTypeKV>();
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void logits_mask_(const uint32_t qo_packed_idx_base,
                                             const uint32_t kv_idx_base, const uint32_t qo_len,
                                             const uint32_t kv_len, const uint32_t kv_end,
                                             const uint_fastdiv num_heads,
                                             typename KTraits::DTypeQKAccum (*s_frag)[4]) {
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  const uint32_t q_idx = (qo_packed_idx_base + warp_idx_in_wg * 16 + lane_idx % 16) / num_heads;

  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        const uint32_t kv_idx = kv_idx_base + warpgroup_idx * (NUM_MMA_KV / 2) * 16 + mma_kv * 32 +
                                lane_idx / 16 * 4 + reg_id;
        const bool mask =
            (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                               : kv_idx >= kv_end));
        s_frag[mma_kv][reg_id] = (mask) ? s_frag[mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV; ++mma_kv) {
#pragma unroll
      for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
        const uint32_t kv_idx = kv_idx_base + mma_kv * 16 + lane_idx / 16 * 4 + reg_id;
        const bool mask =
            (!(KTraits::CAUSAL ? (kv_idx + qo_len > kv_len + q_idx || (kv_idx >= kv_end))
                               : kv_idx >= kv_end));
        s_frag[mma_kv][reg_id] = (mask) ? s_frag[mma_kv][reg_id] : (KTraits::MaskFillValue);
      }
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void update_mdo_states_(typename KTraits::SharedStorage* smem_storage,
                                                   const uint32_t stage_idx,
                                                   typename KTraits::AttentionVariant variant,
                                                   typename KTraits::DTypeQKAccum (*s_frag)[4],
                                                   float (*o_frag)[4],
                                                   typename KTraits::DTypeQKAccum* m, float* d) {
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  const float sm_scale = variant.sm_scale_log2;
  const uint32_t warpgroup_idx = threadIdx.z, lane_idx = threadIdx.x, warp_idx_in_wg = threadIdx.y;
  float m_prev = m[0];
  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      float m_local =
          max(max(s_frag[mma_kv][0], s_frag[mma_kv][1]), max(s_frag[mma_kv][2], s_frag[mma_kv][3]));
      m[0] = max(m[0], m_local);
    }
    m[0] = max(m[0], math::shfl_xor_sync(m[0], 32));
    m[0] = max(m[0], math::shfl_xor_sync(m[0], 16));
    if (lane_idx / 16 == 0) {
      smem_storage->m_wg[warpgroup_idx][warp_idx_in_wg * 16 + lane_idx % 16] = m[0];
    }

    sync_threads();
    // reduce two warpgroups local_max
    m[0] = max(smem_storage->m_wg[0][warp_idx_in_wg * 16 + lane_idx % 16],
               smem_storage->m_wg[1][warp_idx_in_wg * 16 + lane_idx % 16]);
    float o_scale = math::ptx_exp2(m_prev * sm_scale - m[0] * sm_scale);
    d[0] *= o_scale;
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
      fma_f32x2(&o_frag[mma_d][0], &o_frag[mma_d][0], o_scale);
      fma_f32x2(&o_frag[mma_d][2], &o_frag[mma_d][2], o_scale);
    }
    auto m_scale = m[0] * sm_scale * -1;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV / 2; ++mma_kv) {
      // s_frag = exp(s_frag * sm_scale - m * sm_scale)
      fma_f32x2(&s_frag[mma_kv][0], &s_frag[mma_kv][0], sm_scale, m_scale);
      fma_f32x2(&s_frag[mma_kv][2], &s_frag[mma_kv][2], sm_scale, m_scale);
      s_frag[mma_kv][0] = math::ptx_exp2(s_frag[mma_kv][0]);
      s_frag[mma_kv][1] = math::ptx_exp2(s_frag[mma_kv][1]);
      s_frag[mma_kv][2] = math::ptx_exp2(s_frag[mma_kv][2]);
      s_frag[mma_kv][3] = math::ptx_exp2(s_frag[mma_kv][3]);
    }
  } else {
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      float m_local =
          max(max(s_frag[mma_kv][0], s_frag[mma_kv][1]), max(s_frag[mma_kv][2], s_frag[mma_kv][3]));
      m[0] = max(m[0], m_local);
    }
    m[0] = max(m[0], math::shfl_xor_sync(m[0], 32));
    m[0] = max(m[0], math::shfl_xor_sync(m[0], 16));

    float o_scale = math::ptx_exp2(m_prev * sm_scale - m[0] * sm_scale);
    d[0] *= o_scale;
    auto m_scale = m[0] * sm_scale * -1;
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
      fma_f32x2(&o_frag[mma_d][0], &o_frag[mma_d][0], o_scale);
      fma_f32x2(&o_frag[mma_d][2], &o_frag[mma_d][2], o_scale);
    }
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < KTraits::NUM_MMA_KV; ++mma_kv) {
      // s_frag = exp(s_frag * sm_scale - m * sm_scale)
      fma_f32x2(&s_frag[mma_kv][0], &s_frag[mma_kv][0], sm_scale, m_scale);
      fma_f32x2(&s_frag[mma_kv][2], &s_frag[mma_kv][2], sm_scale, m_scale);
      s_frag[mma_kv][0] = math::ptx_exp2(s_frag[mma_kv][0]);
      s_frag[mma_kv][1] = math::ptx_exp2(s_frag[mma_kv][1]);
      s_frag[mma_kv][2] = math::ptx_exp2(s_frag[mma_kv][2]);
      s_frag[mma_kv][3] = math::ptx_exp2(s_frag[mma_kv][3]);
    }
  }
}

template <typename KTraits>
__device__ __forceinline__ void compute_p(typename KTraits::SharedStorage* smem_storage,
                                          const uint32_t stage_idx,
                                          typename KTraits::DTypeQKAccum (*s_frag)[4],
                                          typename KTraits::DTypeQKAccum* d) {
  if constexpr (KTraits::QK_SHARD) {
    const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z,
                   warp_idx_in_wg = threadIdx.y;
    constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
    constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
    // shard s_frag computation on KV dimension across warpgroups, need allgather
    alignas(16) typename KTraits::DTypeKV p_f16[NUM_MMA_KV / 2][4];
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
      vec_cast<typename KTraits::DTypeKV, float>::template cast<4>(p_f16[mma_kv], s_frag[mma_kv]);
      mma::m16k16_rowsum_f16f16f32(d, p_f16[mma_kv]);
    }

    smem_t<KTraits::SWIZZLE_MODE_P> p_smem(smem_storage->kpe_p_smem[stage_idx]);
    constexpr uint32_t UPCAST_STRIDE_P = KTraits::UPCAST_STRIDE_P_64B;
#pragma unroll
    for (uint32_t mma_kv = 0; mma_kv < NUM_MMA_KV / 2; ++mma_kv) {
      uint32_t p_smem_offset_w = p_smem.template get_permuted_offset_64b<UPCAST_STRIDE_P>(
          warp_idx_in_wg * 16 + lane_idx % 16,
          warpgroup_idx * NUM_MMA_KV * 2 + mma_kv * 8 + lane_idx / 16);
      p_smem.store_64b(p_smem_offset_w, (uint32_t*)p_f16[mma_kv]);
    }
    sync_threads();
  }
}

template <typename KTraits>
__device__ __forceinline__ void normalize_d_(typename KTraits::SharedStorage* smem_storage,
                                             const uint32_t stage_idx, float (*o_frag)[4],
                                             typename KTraits::DTypeQKAccum* m, float* d) {
  const uint32_t warpgroup_idx = threadIdx.z, lane_idx = threadIdx.x, warp_idx_in_wg = threadIdx.y;
  if constexpr (KTraits::QK_SHARD) {
#pragma unroll
    for (uint32_t j = 0; j < 1; ++j) {
      if (lane_idx / 16 == 0) {
        smem_storage->d_wg[warpgroup_idx][warp_idx_in_wg * 16 + lane_idx % 16] = d[j];
      }
    }
    sync_threads();
#pragma unroll
    for (uint32_t j = 0; j < 1; ++j) {
      d[j] = smem_storage->d_wg[0][warp_idx_in_wg * 16 + lane_idx % 16] +
             smem_storage->d_wg[1][warp_idx_in_wg * 16 + lane_idx % 16];
    }
  }

  float d_rcp[1];
  // compute reciprocal of d
#pragma unroll
  for (uint32_t j = 0; j < 1; ++j) {
    d_rcp[j] = (m[j] != typename KTraits::DTypeQKAccum(-math::inf)) ? math::ptx_rcp(d[j]) : 0.f;
  }

#pragma unroll
  for (uint32_t mma_d = 0; mma_d < KTraits::NUM_MMA_D_CKV / 2; ++mma_d) {
    fma_f32x2(&o_frag[mma_d][0], &o_frag[mma_d][0], d_rcp[0]);
    fma_f32x2(&o_frag[mma_d][2], &o_frag[mma_d][2], d_rcp[0]);
  }
}

template <typename KTraits>
__device__ __forceinline__ void finalize_m_(typename KTraits::AttentionVariant variant,
                                            typename KTraits::DTypeQKAccum* m) {
  if constexpr (variant.use_softmax) {
#pragma unroll
    for (uint32_t j = 0; j < 1; ++j) {
      if (m[j] != typename KTraits::DTypeQKAccum(-math::inf)) {
        m[j] *= variant.sm_scale_log2;
      }
    }
  }
}

template <typename KTraits>
__device__ void DevicePersistentMergeStates(
    typename KTraits::IdType* merge_packed_offset_start,
    typename KTraits::IdType* merge_packed_offset_end,
    typename KTraits::IdType* merge_partial_packed_offset_start,
    typename KTraits::IdType* merge_partial_packed_offset_end,
    typename KTraits::IdType* merge_partial_stride, typename KTraits::DTypeO* partial_o,
    float* partial_lse, typename KTraits::DTypeO* final_o, float* final_lse,
    const uint32_t o_stride_n, const uint32_t o_stride_h, const uint_fastdiv& num_heads) {
  constexpr uint32_t VEC_SIZE = 8;  // partial o has data type float
  constexpr uint32_t NUM_THRS_PER_ROW = KTraits::HEAD_DIM_CKV / VEC_SIZE;
  constexpr uint32_t ROWS_PER_ITERATION = (KTraits::NUM_THREADS) / NUM_THRS_PER_ROW;
  const uint32_t cta_idx = (gridDim.x * blockIdx.y + blockIdx.x);
  const uint32_t thread_id = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  const uint32_t offset_start = merge_packed_offset_start[cta_idx];
  const uint32_t len = merge_packed_offset_end[cta_idx] - offset_start;
  const uint32_t partial_offset_start = merge_partial_packed_offset_start[cta_idx];
  const uint32_t partial_offset_end = merge_partial_packed_offset_end[cta_idx];
  const uint32_t stride = merge_partial_stride[cta_idx];

  for (uint32_t local_packed_offset = thread_id / NUM_THRS_PER_ROW; local_packed_offset < len;
       local_packed_offset += ROWS_PER_ITERATION) {
    uint32_t final_packed_offset = offset_start + local_packed_offset;
    uint32_t q, r;
    num_heads.divmod(final_packed_offset, q, r);
    state_t<VEC_SIZE> st;

    for (uint32_t partial_packed_offset = partial_offset_start + local_packed_offset;
         partial_packed_offset < partial_offset_end; partial_packed_offset += stride) {
      vec_t<float, VEC_SIZE> o_partial;
      float lse_partial;
      o_partial.cast_load(partial_o + partial_packed_offset * KTraits::HEAD_DIM_CKV +
                          (thread_id % NUM_THRS_PER_ROW) * VEC_SIZE);
      lse_partial = partial_lse[partial_packed_offset];
      st.merge(o_partial, lse_partial, 1);
    }
    st.normalize();
    st.o.cast_store(final_o +
                    (q * o_stride_n + r * o_stride_h + (thread_id % NUM_THRS_PER_ROW) * VEC_SIZE));
    if (final_lse) {
      final_lse[q * num_heads + r] = st.get_lse();
    }
  }
}

template <typename KTraits, bool LDS_TRANS_ENABLE = false>
__device__ __forceinline__ void write_o(typename KTraits::SharedStorage* smem_storage,
                                        typename KTraits::DTypeO* final_o, float* final_lse,
                                        typename KTraits::DTypeO* partial_o, float* partial_lse,
                                        float (*o_frag)[4], typename KTraits::DTypeQKAccum* m,
                                        float* d, const uint32_t o_stride_n,
                                        const uint32_t o_stride_h, const uint32_t q_len,
                                        const uint32_t packed_offset,
                                        const uint_fastdiv& num_heads) {
  using DTypeO = typename KTraits::DTypeO;
  constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  constexpr uint32_t HEAD_DIM_CKV = KTraits::HEAD_DIM_CKV;
  constexpr uint32_t UPCAST_STRIDE_FINAL_O = KTraits::UPCAST_STRIDE_FINAL_O;
  constexpr uint32_t UPCAST_STRIDE_FINAL_O_64B = KTraits::UPCAST_STRIDE_FINAL_O_64B;
  constexpr uint32_t TILE_RATIO = KTraits::CTA_TILE_Q / 16;
  const uint32_t lane_idx = threadIdx.x, warpgroup_idx = threadIdx.z, warp_idx_in_wg = threadIdx.y;
  smem_t<KTraits::SWIZZLE_MODE_O> o_smem(smem_storage->o_smem);

  static_assert(sizeof(DTypeO) == 2);

  if constexpr (LDS_TRANS_ENABLE) {
    uint32_t o_frag_f16[2];
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2; ++mma_d) {
      vec_cast<DTypeO, float>::template cast<4>((DTypeO*)o_frag_f16, o_frag[mma_d]);
      uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O_64B, 16>(
          warp_idx_in_wg * 16 + lane_idx % 16,
          warpgroup_idx * UPCAST_STRIDE_FINAL_O_64B / 2 + mma_d * 4 + lane_idx / 16);
      o_smem.store_64b(o_smem_offset_w, o_frag_f16);
    }

    if (partial_o != nullptr) {
// write to partial_o
#pragma unroll
      for (uint32_t j = 0; j < 1; ++j) {
        uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + lane_idx % 16) / num_heads;
        if (lane_idx / 16 == 0 && q_idx < q_len) {
          partial_lse[(blockIdx.x * TILE_RATIO + warp_idx_in_wg) * 16 + lane_idx % 16] =
              math::ptx_log2(d[j]) + float(m[j]);
        }
      }

      sync_threads();

#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + 4 * j + lane_idx / 16) / num_heads;
        DTypeO* o_partial_ptr =
            partial_o +
            ((blockIdx.x * TILE_RATIO + warp_idx_in_wg) * 16 + 4 * j + lane_idx / 16) *
                HEAD_DIM_CKV +
            warpgroup_idx * (HEAD_DIM_CKV / 2) + (lane_idx % 16) * upcast_size_64b<DTypeO>();
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 8; ++mma_d) {
          if (q_idx < q_len) {
            uint32_t o_smem_offset_r =
                o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O_64B, 16>(
                    warp_idx_in_wg * 16 + 4 * j + lane_idx / 16,
                    warpgroup_idx * NUM_MMA_D_CKV * 2 + mma_d * 16 + lane_idx % 16);
            o_smem.load_64b(o_smem_offset_r, o_frag_f16);
            cp_async::store_64b_pred(o_frag_f16, o_partial_ptr, true);
          }
          o_partial_ptr += 16 * upcast_size_64b<DTypeO>();
        }
      }
    } else {
      // write to final_o

      if (final_lse) {
#pragma unroll
        for (uint32_t j = 0; j < 1; ++j) {
          uint32_t q, r;
          num_heads.divmod(packed_offset + j * 32 + warp_idx_in_wg * 16 + lane_idx % 16, q, r);
          if (lane_idx / 16 == 0 && q < q_len) {
            final_lse[q * num_heads + r] = math::ptx_log2(d[j]) + float(m[j]);
          }
        }
      }

      sync_threads();

#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t q, r;
        num_heads.divmod(packed_offset + warp_idx_in_wg * 16 + 4 * j + lane_idx / 16, q, r);
        DTypeO* o_final_ptr = final_o + q * o_stride_n + r * o_stride_h +
                              warpgroup_idx * (HEAD_DIM_CKV / 2) +
                              (lane_idx % 16) * upcast_size_64b<DTypeO>();
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 8; ++mma_d) {
          if (q < q_len) {
            uint32_t o_smem_offset_r =
                o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O_64B, 16>(
                    warp_idx_in_wg * 16 + 4 * j + lane_idx / 16,
                    warpgroup_idx * NUM_MMA_D_CKV * 2 + mma_d * 16 + lane_idx % 16);
            o_smem.load_64b(o_smem_offset_r, o_frag_f16);
            cp_async::store_64b_pred(o_frag_f16, o_final_ptr, true);
          }
          o_final_ptr += 16 * upcast_size_64b<DTypeO>();
        }
      }
    }
  } else {
    float* o_frag_flatten = &o_frag[0][0];

    if constexpr (KTraits::CTA_TILE_Q == 64) {
      // used for lds_b64x4(CKV_USE_64B)
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 2 / 4; ++mma_d) {
#pragma unroll
        for (uint32_t col = 0; col < 2; ++col) {
          uint32_t o_frag_f16[8 / 2];
          float o_frag_f32[8];
          o_frag_f32[0] = o_frag_flatten[mma_d * 16 + col * 2 + 0];
          o_frag_f32[1] = o_frag_flatten[mma_d * 16 + col * 2 + 4];
          o_frag_f32[2] = o_frag_flatten[mma_d * 16 + col * 2 + 8];
          o_frag_f32[3] = o_frag_flatten[mma_d * 16 + col * 2 + 12];
          o_frag_f32[4] = o_frag_flatten[mma_d * 16 + col * 2 + 1];
          o_frag_f32[5] = o_frag_flatten[mma_d * 16 + col * 2 + 5];
          o_frag_f32[6] = o_frag_flatten[mma_d * 16 + col * 2 + 9];
          o_frag_f32[7] = o_frag_flatten[mma_d * 16 + col * 2 + 13];
          vec_cast<DTypeO, float>::template cast<8>((DTypeO*)o_frag_f16, o_frag_f32);

          uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
              warp_idx_in_wg * 16 + lane_idx % 16,
              warpgroup_idx * UPCAST_STRIDE_FINAL_O / 2 + mma_d * 8 + lane_idx / 16 * 2 + col);
          o_smem.store_128b(o_smem_offset_w, o_frag_f16);
        }
      }
    } else {  // KTraits::CTA_TILE_Q == 32
              // used for lds_b128x4(CKV_USE_128B)
#pragma unroll
      for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 16; ++mma_d) {
#pragma unroll
        for (uint32_t col = 0; col < 4; ++col) {
          uint32_t o_frag_f16[8 / 2];
          float o_frag_f32[8];
#pragma unroll
          for (size_t i = 0; i < 8; ++i) {
            o_frag_f32[i] = o_frag_flatten[mma_d * 32 + col + i * 4];
          }
          vec_cast<DTypeO, float>::template cast<8>((DTypeO*)o_frag_f16, o_frag_f32);

          uint32_t o_smem_offset_w = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
              warp_idx_in_wg * 16 + lane_idx % 16,
              warpgroup_idx * UPCAST_STRIDE_FINAL_O / 2 + mma_d * 16 + lane_idx / 16 * 4 + col);
          o_smem.store_128b(o_smem_offset_w, o_frag_f16);
        }
      }
    }

    if (partial_o != nullptr) {
// write to partial_o
#pragma unroll
      for (uint32_t j = 0; j < 1; ++j) {
        uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + lane_idx % 16) / num_heads;
        if (lane_idx / 16 == 0 && q_idx < q_len) {
          partial_lse[(blockIdx.x * TILE_RATIO + warp_idx_in_wg) * 16 + lane_idx % 16] =
              math::ptx_log2(d[j]) + float(m[j]);
        }
      }

      sync_threads();

#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q_idx = (packed_offset + warp_idx_in_wg * 16 + 8 * j + lane_idx / 8) / num_heads;
        DTypeO* o_partial_ptr =
            partial_o +
            ((blockIdx.x * TILE_RATIO + warp_idx_in_wg) * 16 + 8 * j + lane_idx / 8) *
                HEAD_DIM_CKV +
            warpgroup_idx * (HEAD_DIM_CKV / 2) + (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 8; ++mma_d) {
          if (q_idx < q_len) {
            uint32_t o_frag_f16[8 / 2];
            uint32_t o_smem_offset_r = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
                warp_idx_in_wg * 16 + 8 * j + lane_idx / 8,
                warpgroup_idx * NUM_MMA_D_CKV + mma_d * 8 + lane_idx % 8);
            o_smem.load_128b(o_smem_offset_r, o_frag_f16);
            cp_async::store_128b_pred(o_frag_f16, o_partial_ptr, true);
          }
          o_partial_ptr += 8 * upcast_size<DTypeO>();
        }
      }
    } else {
      // write to final_o

      if (final_lse) {
#pragma unroll
        for (uint32_t j = 0; j < 1; ++j) {
          uint32_t q, r;
          num_heads.divmod(packed_offset + j * 32 + warp_idx_in_wg * 16 + lane_idx % 16, q, r);
          if (lane_idx / 16 == 0 && q < q_len) {
            final_lse[q * num_heads + r] = math::ptx_log2(d[j]) + float(m[j]);
          }
        }
      }

      sync_threads();

#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        uint32_t q, r;
        num_heads.divmod(packed_offset + warp_idx_in_wg * 16 + 8 * j + lane_idx / 8, q, r);
        DTypeO* o_final_ptr = final_o + q * o_stride_n + r * o_stride_h +
                              warpgroup_idx * (HEAD_DIM_CKV / 2) +
                              (lane_idx % 8) * upcast_size<DTypeO>();
#pragma unroll
        for (uint32_t mma_d = 0; mma_d < NUM_MMA_D_CKV / 8; ++mma_d) {
          if (q < q_len) {
            uint32_t o_frag_f16[8 / 2];
            uint32_t o_smem_offset_r = o_smem.template get_permuted_offset<UPCAST_STRIDE_FINAL_O>(
                warp_idx_in_wg * 16 + 8 * j + lane_idx / 8,
                warpgroup_idx * NUM_MMA_D_CKV + mma_d * 8 + lane_idx % 8);
            o_smem.load_128b(o_smem_offset_r, o_frag_f16);
            cp_async::store_128b_pred(o_frag_f16, o_final_ptr, true);
          }
          o_final_ptr += 8 * upcast_size<DTypeO>();
        }
      }
    }
  }
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_FA2_UTILS_BASE_CUH_