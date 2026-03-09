/*
 * Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 */
#ifndef FLASHINFER_MLA_KERNELS_XCORE1000_CUH_
#define FLASHINFER_MLA_KERNELS_XCORE1000_CUH_

#include "mla_utils_base.cuh"

namespace flashinfer {

namespace mla {

template <typename KTraits, typename Params>
__device__ __forceinline__ void batch_mla_paged_attention_kernel_xc1000_ctq64(const Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  extern __shared__ __align__(alignof(typename KTraits::SharedStorage)) uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  typename KTraits::AttentionVariant variant(params, blockIdx.y, smem);

  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = KTraits::SWIZZLE_MODE_Q_NOPE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_PE = KTraits::SWIZZLE_MODE_Q_PE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_CKV = KTraits::SWIZZLE_MODE_CKV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KPE = KTraits::SWIZZLE_MODE_KPE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q_PER_WAVE = KTraits::NUM_MMA_Q_PER_WAVE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::CAUSAL;

  DTypeQ* q_nope = params.q_nope;
  DTypeQ* q_pe = params.q_pe;
  DTypeKV* ckv = params.ckv;
  DTypeKV* kpe = params.kpe;
  IdType* kv_indices = params.kv_indices;
  DTypeO* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  float s_frag[NUM_MMA_KV_PER_WAVE][4];
  alignas(16) float o_frag[NUM_MMA_D_CKV / 2][4];
  float m[NUM_MMA_Q_PER_WAVE];
  float d[NUM_MMA_Q_PER_WAVE];

  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t q_nope_stride_n = params.q_nope_stride_n;
  const uint32_t q_nope_stride_h = params.q_nope_stride_h;
  const uint32_t q_pe_stride_n = params.q_pe_stride_n;
  const uint32_t q_pe_stride_h = params.q_pe_stride_h;
  const uint32_t ckv_stride_page = params.ckv_stride_page;
  const uint32_t ckv_stride_n = params.ckv_stride_n;
  const uint32_t kpe_stride_page = params.kpe_stride_page;
  const uint32_t kpe_stride_n = params.kpe_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    constexpr uint32_t mma_kv_num = (KTraits::CTA_TILE_Q == 32) ? NUM_MMA_KV : NUM_MMA_KV / 2;
    uint32_t q_nope_frag[NUM_MMA_Q_PER_WAVE][NUM_MMA_D_CKV][2];
    uint32_t q_rope_frag[NUM_MMA_Q_PER_WAVE][NUM_MMA_D_KPE][2];
    uint32_t ckv_frag[mma_kv_num][NUM_MMA_D_CKV / 4][2];
    uint32_t kpe_frag[mma_kv_num][NUM_MMA_D_KPE / 4][2];

    const uint32_t q_indptr = params.q_indptr[work_idx];
    const uint32_t kv_indptr = params.kv_indptr[work_idx];
    const int32_t partial_indptr = params.partial_indptr[work_idx];
    const uint32_t q_len = params.q_len[work_idx];
    const uint32_t kv_len = params.kv_len[work_idx];
    const uint32_t packed_qo_start = params.q_start[work_idx];
    const uint32_t kv_start = params.kv_start[work_idx];
    const uint32_t kv_end = params.kv_end[work_idx];

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
    const uint32_t qo_upperbound =
        min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

    uint32_t k_offset_r[4];
    uint32_t kpe_offset_r[4];
    uint32_t v_offset_r[4];
    get_k_base_offset_r<KTraits>(&smem_storage, k_offset_r, kpe_offset_r);
    get_v_base_offset_r<KTraits>(&smem_storage, v_offset_r);

    init_states_<KTraits>(o_frag, m, d);

    sync_threads();

    load_q_partial<KTraits, KTraits::UPCAST_STRIDE_Q_PE, KTraits::NUM_MMA_D_KPE>(
        &smem_storage, q_pe + q_indptr * q_pe_stride_n, q_pe_stride_n, q_pe_stride_h, qo_upperbound,
        qo_packed_idx_base, params.num_heads);
    sync_threads();
    load_q_smem_reg_pe<KTraits, NUM_MMA_D_KPE>(&smem_storage, q_rope_frag);

    int kv_tile_idx =
        ceil_div(
            (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                    : kv_end),
            CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    uint32_t block_iter_base = kv_indptr * block_size + kv_start;
    sync_threads();
    uint32_t kv_page_idx[mma_kv_num];
    // 0 <= kv_page_offset < page_size, so kv_page_offset always equals 0 when page_size = 1
    uint32_t kv_page_offset[mma_kv_num];
    int64_t ckv_offset[NUM_MMA_KV_PER_WAVE];
    int64_t kpe_offset[NUM_MMA_KV_PER_WAVE];

    // last kv tile, only last kv tile Is_even_MN should be false
    uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;
    prefetch_kv_indices_64b<KTraits, /*Is_even_MN=*/false>(
        block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size, packed_kv_bound, kv_indices,
        ckv_offset, kpe_offset, ckv_stride_n, ckv_stride_page, kpe_stride_n, kpe_stride_page);

    int mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / num_heads) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    load_q_partial<KTraits, KTraits::UPCAST_STRIDE_Q_NOPE, KTraits::NUM_MMA_D_CKV>(
        &smem_storage, q_nope + q_indptr * q_nope_stride_n, q_nope_stride_n, q_nope_stride_h,
        qo_upperbound, qo_packed_idx_base, params.num_heads);
    sync_threads();
    load_q_smem_reg_nope<KTraits, NUM_MMA_D_CKV>(&smem_storage, q_nope_frag);

    load_kv_r<KTraits, /*Is_even_MN=*/false>(ckv, kpe, ckv_frag, kpe_frag, ckv_offset, kpe_offset,
                                             packed_kv_bound,
                                             block_iter_base + kv_tile_idx * CTA_TILE_KV);

    // loop with mask
#pragma unroll 1
    for (; kv_tile_idx >= mask_tile_idx && kv_tile_idx > 0; --kv_tile_idx) {
      clear<float, 4 * NUM_MMA_KV_PER_WAVE>(s_frag[0]);
      sync_threads();
      prefetch_kv_indices_64b<KTraits, /*Is_even_MN=*/true>(
          block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV, block_size, packed_kv_bound,
          kv_indices, kv_page_idx, kv_page_offset);
      load_kv_w<KTraits>(&smem_storage, ckv_frag, kpe_frag, kv_tile_idx % NUM_STAGES);
      compute_kv_offset_64b<KTraits>(kv_page_idx, kv_page_offset, ckv_offset, kpe_offset,
                                     ckv_stride_n, ckv_stride_page, kpe_stride_n, kpe_stride_page);
      sync_threads();
      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frag, q_rope_frag,
                              s_frag, k_offset_r, kpe_offset_r);

      // load k_pe
      load_kv_r<KTraits, KTraits::NUM_MMA_D_KPE, 0, KTraits::NUM_MMA_D_KPE / 4,
                /*Is_even_MN=*/true>(kpe, kpe_frag, kpe_offset, packed_kv_bound,
                                     block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // logits mask
      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // load kv_ne_1-4
      load_kv_r<KTraits, KTraits::NUM_MMA_D_CKV, 0, KTraits::NUM_MMA_D_CKV / 8,
                /*Is_even_MN=*/true>(ckv, ckv_frag, ckv_offset, packed_kv_bound,
                                     block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      compute_p<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d);

      // load kv_ne_5-8
      load_kv_r<KTraits, KTraits::NUM_MMA_D_CKV, KTraits::NUM_MMA_D_CKV / 8,
                KTraits::NUM_MMA_D_CKV / 4, /*Is_even_MN=*/true>(
          ckv, ckv_frag, ckv_offset, packed_kv_bound,
          block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag,
                              v_offset_r);
    }

    // loop without mask
#pragma unroll 1
    for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
      clear<float, 4 * NUM_MMA_KV_PER_WAVE>(s_frag[0]);
      sync_threads();

      prefetch_kv_indices_64b<KTraits, /*Is_even_MN=*/true>(
          block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV, block_size, packed_kv_bound,
          kv_indices, kv_page_idx, kv_page_offset);
      load_kv_w<KTraits>(&smem_storage, ckv_frag, kpe_frag, kv_tile_idx % NUM_STAGES);

      compute_kv_offset_64b<KTraits>(kv_page_idx, kv_page_offset, ckv_offset, kpe_offset,
                                     ckv_stride_n, ckv_stride_page, kpe_stride_n, kpe_stride_page);
      sync_threads();
      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frag, q_rope_frag,
                              s_frag, k_offset_r, kpe_offset_r);

      // load kv_ne_1-4
      load_kv_r<KTraits, KTraits::NUM_MMA_D_CKV, 0, KTraits::NUM_MMA_D_CKV / 8,
                /*Is_even_MN=*/true>(ckv, ckv_frag, ckv_offset, packed_kv_bound,
                                     block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      // load kv_ne_5-8
      load_kv_r<KTraits, KTraits::NUM_MMA_D_CKV, KTraits::NUM_MMA_D_CKV / 8,
                KTraits::NUM_MMA_D_CKV / 4, /*Is_even_MN=*/true>(
          ckv, ckv_frag, ckv_offset, packed_kv_bound,
          block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      compute_p<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d);

      // load k_pe
      load_kv_r<KTraits, KTraits::NUM_MMA_D_KPE, 0, KTraits::NUM_MMA_D_KPE / 4,
                /*Is_even_MN=*/true>(kpe, kpe_frag, kpe_offset, packed_kv_bound,
                                     block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag,
                              v_offset_r);
    }
    sync_threads();

    // last tiles
    for (; kv_tile_idx >= 0; --kv_tile_idx) {
      clear<float, 4 * NUM_MMA_KV_PER_WAVE>(s_frag[0]);
      load_kv_w<KTraits>(&smem_storage, ckv_frag, kpe_frag, kv_tile_idx % NUM_STAGES);
      sync_threads();
      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frag, q_rope_frag,
                              s_frag, k_offset_r, kpe_offset_r);

      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      compute_p<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag,
                              v_offset_r);
    }

    sync_threads();

    // normalize and write back
    normalize_d_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, o_frag, m, d);

    finalize_m_<KTraits>(variant, m);

    write_o<KTraits>(
        &smem_storage, final_o + q_indptr * o_stride_n,
        final_lse ? final_lse + q_indptr * num_heads : nullptr,
        (partial_indptr == -1) ? nullptr : partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
        (partial_indptr == -1) ? nullptr : partial_lse + partial_indptr, o_frag, m, d, o_stride_n,
        o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads);
  }

  auto grid = cg::this_grid();
  grid.sync();

  // the second stage, merge partial outputs
  DevicePersistentMergeStates<KTraits>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, partial_o, partial_lse, final_o, final_lse, o_stride_n,
      o_stride_h, num_heads);
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void batch_mla_paged_attention_kernel_xc1000_ctq32(const Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;

  extern __shared__ __align__(alignof(typename KTraits::SharedStorage)) uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);

  typename KTraits::AttentionVariant variant(params, blockIdx.y, smem);

  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_NOPE = KTraits::SWIZZLE_MODE_Q_NOPE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q_PE = KTraits::SWIZZLE_MODE_Q_PE;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_CKV = KTraits::SWIZZLE_MODE_CKV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KPE = KTraits::SWIZZLE_MODE_KPE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV_PER_WAVE = KTraits::NUM_MMA_KV_PER_WAVE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q_PER_WAVE = KTraits::NUM_MMA_Q_PER_WAVE;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_CKV = KTraits::NUM_MMA_D_CKV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_KPE = KTraits::NUM_MMA_D_KPE;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr int32_t NUM_STAGES = KTraits::NUM_STAGES;
  [[maybe_unused]] constexpr bool CAUSAL = KTraits::CAUSAL;

  DTypeQ* q_nope = params.q_nope;
  DTypeQ* q_pe = params.q_pe;
  DTypeKV* ckv = params.ckv;
  DTypeKV* kpe = params.kpe;
  IdType* kv_indices = params.kv_indices;
  DTypeO* partial_o = params.partial_o;
  float* partial_lse = params.partial_lse;
  DTypeO* final_o = params.final_o;
  float* final_lse = params.final_lse;
  IdType* work_indptr = params.work_indptr;

  float s_frag[NUM_MMA_KV_PER_WAVE][4];
  alignas(16) float o_frag[NUM_MMA_D_CKV / 2][4];
  float m[NUM_MMA_Q_PER_WAVE];
  float d[NUM_MMA_Q_PER_WAVE];

  const uint_fastdiv& num_heads = params.num_heads;
  const uint_fastdiv& block_size = params.block_size;
  const uint32_t q_nope_stride_n = params.q_nope_stride_n;
  const uint32_t q_nope_stride_h = params.q_nope_stride_h;
  const uint32_t q_pe_stride_n = params.q_pe_stride_n;
  const uint32_t q_pe_stride_h = params.q_pe_stride_h;
  const uint32_t ckv_stride_page = params.ckv_stride_page;
  const uint32_t ckv_stride_n = params.ckv_stride_n;
  const uint32_t kpe_stride_page = params.kpe_stride_page;
  const uint32_t kpe_stride_n = params.kpe_stride_n;
  const uint32_t o_stride_n = params.o_stride_n;
  const uint32_t o_stride_h = params.o_stride_h;
  const uint32_t cluster_tile_q = gridDim.x * KTraits::CTA_TILE_Q;

#pragma unroll 1
  for (IdType work_idx = work_indptr[blockIdx.y]; work_idx < work_indptr[blockIdx.y + 1];
       ++work_idx) {
    constexpr uint32_t mma_kv_num = NUM_MMA_KV == 1 ? 1 : NUM_MMA_KV / 2;
    uint32_t q_nope_frag[NUM_MMA_Q_PER_WAVE][NUM_MMA_D_CKV / 2][4];
    uint32_t q_rope_frag[NUM_MMA_Q_PER_WAVE][NUM_MMA_D_KPE / 2][4];
    uint32_t ckv_frag[mma_kv_num][NUM_MMA_D_CKV / 4][4];
    uint32_t kpe_frag[mma_kv_num][NUM_MMA_D_KPE / 4][4];

    const uint32_t q_indptr = params.q_indptr[work_idx];
    const uint32_t kv_indptr = params.kv_indptr[work_idx];
    const int32_t partial_indptr = params.partial_indptr[work_idx];
    const uint32_t q_len = params.q_len[work_idx];
    const uint32_t kv_len = params.kv_len[work_idx];
    const uint32_t packed_qo_start = params.q_start[work_idx];
    const uint32_t kv_start = params.kv_start[work_idx];
    const uint32_t kv_end = params.kv_end[work_idx];

    const uint32_t qo_packed_idx_base = packed_qo_start + blockIdx.x * KTraits::CTA_TILE_Q;
    const uint32_t qo_upperbound =
        min(q_len, ceil_div(qo_packed_idx_base + KTraits::CTA_TILE_Q, num_heads));

    init_states_<KTraits>(o_frag, m, d);

    sync_threads();

    load_q<KTraits>(&smem_storage, q_nope + q_indptr * q_nope_stride_n,
                    q_pe + q_indptr * q_pe_stride_n, q_nope_stride_n, q_nope_stride_h,
                    q_pe_stride_n, q_pe_stride_h, qo_upperbound, qo_packed_idx_base,
                    params.num_heads);
    sync_threads();

    load_q_smem_reg<KTraits, NUM_MMA_D_CKV, NUM_MMA_D_KPE>(&smem_storage, q_nope_frag, q_rope_frag);

    int kv_tile_idx =
        ceil_div(
            (CAUSAL ? min(kv_end, kv_len - q_len + (packed_qo_start + cluster_tile_q) / num_heads)
                    : kv_end),
            CTA_TILE_KV) -
        1 - (kv_start / CTA_TILE_KV);

    uint32_t block_iter_base = kv_indptr * block_size + kv_start;
    sync_threads();
    uint32_t kv_page_idx[mma_kv_num];
    // 0 <= kv_page_offset < page_size, so kv_page_offset always equals 0 when page_size = 1
    uint32_t kv_page_offset[mma_kv_num];
    int64_t ckv_offset[NUM_MMA_KV_PER_WAVE];
    int64_t kpe_offset[NUM_MMA_KV_PER_WAVE];

    // last kv tile, only last kv tile Is_even_MN should be false
    uint32_t packed_kv_bound = kv_indptr * block_size + kv_len;
    prefetch_kv_indices<KTraits, /*Is_even_MN=*/false>(
        block_iter_base + kv_tile_idx * CTA_TILE_KV, block_size, packed_kv_bound, kv_indices,
        ckv_offset, kpe_offset, ckv_stride_n, ckv_stride_page, kpe_stride_n, kpe_stride_page);

    int mask_tile_idx =
        (CAUSAL ? min(kv_end, kv_len - q_len + packed_qo_start / num_heads) : kv_end) /
            CTA_TILE_KV -
        (kv_start / CTA_TILE_KV);

    load_kv_r<KTraits, /*Is_even_MN=*/false>(ckv, kpe, ckv_frag, kpe_frag, ckv_offset, kpe_offset,
                                             packed_kv_bound,
                                             block_iter_base + kv_tile_idx * CTA_TILE_KV);

    load_kv_w<KTraits>(&smem_storage, ckv_frag, kpe_frag, kv_tile_idx % NUM_STAGES);

#pragma unroll 1
    for (; kv_tile_idx + 1 > NUM_STAGES; --kv_tile_idx) {
      clear<float, 4 * NUM_MMA_KV_PER_WAVE>(s_frag[0]);
      sync_threads();
      prefetch_kv_indices<KTraits, /*Is_even_MN=*/true>(
          block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV, block_size, packed_kv_bound,
          kv_indices, ckv_offset, kpe_offset, ckv_stride_n, ckv_stride_page, kpe_stride_n,
          kpe_stride_page);

      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frag, q_rope_frag,
                              s_frag);

      // load k_pe
      load_kv_r<KTraits, KTraits::NUM_MMA_D_KPE, 0, KTraits::NUM_MMA_D_KPE / 4,
                /*Is_even_MN=*/true>(kpe, kpe_frag, kpe_offset, packed_kv_bound,
                                     block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // logits mask
      if (kv_tile_idx >= mask_tile_idx) {
        logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len,
                              kv_len, kv_end, num_heads, s_frag);
      }

      // load kv_ne_1-4
      load_kv_r<KTraits, KTraits::NUM_MMA_D_CKV, 0, KTraits::NUM_MMA_D_CKV / 8,
                /*Is_even_MN=*/true>(ckv, ckv_frag, ckv_offset, packed_kv_bound,
                                     block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);

      compute_p<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d);

      // load kv_ne_5-8
      load_kv_r<KTraits, KTraits::NUM_MMA_D_CKV, KTraits::NUM_MMA_D_CKV / 8,
                KTraits::NUM_MMA_D_CKV / 4, /*Is_even_MN=*/true>(
          ckv, ckv_frag, ckv_offset, packed_kv_bound,
          block_iter_base + (kv_tile_idx - 1) * CTA_TILE_KV);

      // compute sfm * v
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
      sync_threads();
      load_kv_w<KTraits>(&smem_storage, ckv_frag, kpe_frag, kv_tile_idx % NUM_STAGES);
    }

    for (; kv_tile_idx >= 0; --kv_tile_idx) {
      clear<float, 4 * NUM_MMA_KV_PER_WAVE>(s_frag[0]);
      sync_threads();
      // compute mla qk
      compute_mla_qk<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, q_nope_frag, q_rope_frag,
                              s_frag);

      // logits mask
      logits_mask_<KTraits>(qo_packed_idx_base, kv_start + kv_tile_idx * CTA_TILE_KV, q_len, kv_len,
                            kv_end, num_heads, s_frag);

      // compute m,d states in online softmax
      update_mdo_states_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, variant, s_frag, o_frag,
                                  m, d);
      compute_p<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d);
      compute_mla_pv<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, s_frag, d, o_frag);
    }

    sync_threads();

    // normalize and write back
    normalize_d_<KTraits>(&smem_storage, kv_tile_idx % NUM_STAGES, o_frag, m, d);

    finalize_m_<KTraits>(variant, m);

    write_o<KTraits>(
        &smem_storage, final_o + q_indptr * o_stride_n,
        final_lse ? final_lse + q_indptr * num_heads : nullptr,
        (partial_indptr == -1) ? nullptr : partial_o + partial_indptr * KTraits::HEAD_DIM_CKV,
        (partial_indptr == -1) ? nullptr : partial_lse + partial_indptr, o_frag, m, d, o_stride_n,
        o_stride_h, qo_upperbound, qo_packed_idx_base, num_heads);
  }

  auto grid = cg::this_grid();
  grid.sync();

  // the second stage, merge partial outputs
  DevicePersistentMergeStates<KTraits>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, partial_o, partial_lse, final_o, final_lse, o_stride_n,
      o_stride_h, num_heads);
}

}  // namespace mla

}  // namespace flashinfer

#endif  // FLASHINFER_MLA_KERNELS_XCORE1000_CUH_
