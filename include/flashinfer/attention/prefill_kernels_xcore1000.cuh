/*
 * Copyright (c) 2025 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.
 */
#ifndef FLASHINFER_PREFILL_KERNELS_XCORE1000_CUH_
#define FLASHINFER_PREFILL_KERNELS_XCORE1000_CUH_

#include "prefill_utils.cuh"

namespace flashinfer {

template <typename KTraits, typename Params>
__device__ __forceinline__ void batch_prefill_with_ragged_kv_cache_kernel_xc1000(
    const Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t K_THR_LAYOUT_ROW = KTraits::K_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t K_THR_LAYOUT_COL = KTraits::K_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr uint32_t V_THR_LAYOUT_ROW = KTraits::V_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t V_THR_LAYOUT_COL = KTraits::V_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

  using Selector = DeviceFunctionSelector<CTA_TILE_KV, false, KTraits>;
  constexpr auto write_o_reg_gmem_ = Selector::Write_O_Func;
  constexpr auto produce_v_w_ = Selector::Write_V_Func;
  constexpr auto produce_v_r_ = Selector::Read_V_Func;
  constexpr auto compute_sfm_v_ = Selector::Sfm_V_Func;

  DTypeQ* q = params.q;
  IdType* request_indices = params.request_indices;
  IdType* qo_tile_indices = params.qo_tile_indices;
  IdType* kv_tile_indices = params.kv_tile_indices;
  IdType* q_indptr = params.q_indptr;
  IdType* kv_indptr = params.kv_indptr;
  DTypeKV* k = params.k;
  DTypeKV* v = params.v;
  IdType* o_indptr = params.o_indptr;
  DTypeO* o = params.o;
  float* lse = params.lse;
  bool* block_valid_mask = params.block_valid_mask;
  const bool partition_kv = params.partition_kv;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t k_stride_n = params.k_stride_n;
  const uint32_t k_stride_h = params.k_stride_h;
  const uint32_t v_stride_n = params.v_stride_n;
  const uint32_t v_stride_h = params.v_stride_h;
  const int32_t maybe_window_left = params.window_left;
  const uint_fastdiv& group_size = params.group_size;

  static_assert(sizeof(DTypeQ) == 2);
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

  const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<KTraits>();
  uint32_t bx, num_kv_heads, kv_head_idx;
  if constexpr (NUM_MMA_D_QK > NUM_MMA_D_VO) {
    bx = gridDim.z - blockIdx.z - 1;
    num_kv_heads = gridDim.x;
    kv_head_idx = blockIdx.x;
  } else {
    bx = blockIdx.x;
    num_kv_heads = gridDim.z;
    kv_head_idx = blockIdx.z;
  }

  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }

  const uint32_t num_qo_heads = group_size * num_kv_heads;
  const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
  const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                 window_left = variant.window_left;
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t qo_upper_bound = min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

  uint32_t q_frag[NUM_MMA_Q][NUM_MMA_D_QK / 2][4];
  DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][4];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][4];
  DTypeQKAccum m[NUM_MMA_Q];
  float d[NUM_MMA_Q];
  float rope_freq[NUM_MMA_D_QK / 2][4];
  uint32_t k_frag[NUM_MMA_KV * 2 / NUM_WARPS_Q]
                 [KTraits::NUM_MMA_D_QK / (8 / sizeof(typename KTraits::DTypeKV))][4];

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;
    init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta);
  }
  init_states<KTraits>(variant, o_frag, m, d);

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>()) * NUM_MMA_Q * 16;
  smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

  DTypeQ* q_ptr_base =
      q + q_indptr[request_idx] * q_stride_n + kv_head_idx * group_size * q_stride_h;

  DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h
                                    : o + o_indptr[request_idx] * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h;

  uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
      get_warp_idx_q<KTraits>() * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                              q_stride_h, group_size, &qo_smem);

  const uint32_t num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size, chunk_start))
           : chunk_size),
      CTA_TILE_KV);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                      qo_len + window_left + chunk_start),
               CTA_TILE_KV);

  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                        chunk_start))
           : chunk_size) /
      CTA_TILE_KV;

  smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);

  uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
           v_smem_offset_r = v_smem.template get_64bx4_offset<UPCAST_STRIDE_V_64B>(
               get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + lane_idx / 16, lane_idx % 16),
           k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               warp_idx * K_THR_LAYOUT_ROW + lane_idx / K_THR_LAYOUT_COL,
               lane_idx % K_THR_LAYOUT_COL),
           v_smem_offset_w = v_smem.template get_64bx4_offset<UPCAST_STRIDE_V>(
               warp_idx * V_THR_LAYOUT_ROW + lane_idx / V_THR_LAYOUT_COL,
               lane_idx % V_THR_LAYOUT_COL * 2);

  DTypeKV* k_ptr = k +
                   (kv_indptr[request_idx] + chunk_start + warp_idx * K_THR_LAYOUT_ROW +
                    lane_idx / K_THR_LAYOUT_COL) *
                       k_stride_n +
                   kv_head_idx * k_stride_h +
                   (lane_idx % K_THR_LAYOUT_COL) * upcast_size<DTypeKV>();

  produce_k_r<KTraits>(&k_ptr, k_stride_n, 0, chunk_size, k_frag);

  DTypeKV* v_ptr = v +
                   (kv_indptr[request_idx] + chunk_start + warp_idx * V_THR_LAYOUT_ROW * 4 +
                    lane_idx / V_THR_LAYOUT_COL * 4) *
                       v_stride_n +
                   kv_head_idx * v_stride_h +
                   (lane_idx % V_THR_LAYOUT_COL) * upcast_size_64b<DTypeKV>();

  if constexpr (CTA_TILE_KV == 32) {
    v_smem_offset_w =
        (warp_idx * V_THR_LAYOUT_ROW + lane_idx / V_THR_LAYOUT_COL) * UPCAST_STRIDE_V +
        lane_idx % V_THR_LAYOUT_COL;
    v_ptr = v +
            (kv_indptr[request_idx] + chunk_start + warp_idx * V_THR_LAYOUT_ROW +
             lane_idx / V_THR_LAYOUT_COL) *
                v_stride_n +
            kv_head_idx * v_stride_h + (lane_idx % V_THR_LAYOUT_COL) * upcast_size<DTypeKV>();
  }

  auto& v_frag = [NUM_MMA_KV, NUM_WARPS_Q, NUM_MMA_D_VO]() -> auto& {
    if constexpr (CTA_TILE_KV == 32) {
      uint32_t arr[NUM_MMA_KV * 2 / NUM_WARPS_Q * NUM_MMA_D_VO / (8 / sizeof(DTypeKV)) * 4] = {};
      return arr;
    } else {
      uint32_t arr[NUM_MMA_KV / NUM_WARPS_Q * NUM_MMA_D_VO / (8 / sizeof(DTypeKV)) * 4 * 2] = {};
      return arr;
    }
  }();
  sync_threads();

  load_q_smem_reg<KTraits>(&qo_smem, &q_smem_offset_r, q_frag);

  produce_v_r_(&v_ptr, v_stride_n, 0, chunk_size, v_frag);

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    sync_threads();
    IdType* q_rope_offset = nullptr;

    if constexpr (has_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.maybe_q_rope_offset;
    }
    if (!q_rope_offset) {
      q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                           &q_smem_offset_r, rope_freq);
    } else {
      q_smem_inplace_apply_rotary_with_pos<KTraits>(qo_packed_idx_base,
                                                    q_rope_offset + q_indptr[request_idx], &qo_smem,
                                                    group_size, &q_smem_offset_r, rope_freq);
    }
    sync_threads();
  }

  sync_threads();
  produce_k_w<KTraits>(k_smem, &k_smem_offset_w, k_frag);

  if constexpr (KTraits::NUM_MMA_D_QK != KTraits::NUM_MMA_D_VO) {
    __builtin_mxc_igroup_config(1, -1, -1, -1, -1, -1, -1, -1);
    uint32_t k_r_frag[KTraits::NUM_MMA_D_QK / 2][KTraits::NUM_MMA_KV][4];
    uint32_t v_r_frag[KTraits::NUM_MMA_KV][KTraits::NUM_MMA_D_VO / 4][4][2];
    sync_threads();

    produce_k_r<KTraits>(&k_ptr, k_stride_n, CTA_TILE_KV, chunk_size, k_frag);

    lds_k<KTraits>(&k_smem, &k_smem_offset_r, k_r_frag);

    produce_v_w_(v_smem, &v_smem_offset_w, v_frag);

#pragma unroll 1
    for (uint32_t iter = 0; iter < mask_iteration; ++iter) {
      clear<DTypeQKAccum, NUM_MMA_Q * NUM_MMA_KV * 4>(s_frag[0][0]);

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        IdType* k_rope_offset = nullptr;
        if constexpr (has_maybe_k_rope_offset_v<Params>) {
          k_rope_offset = params.maybe_k_rope_offset;
        }
        k_smem_inplace_apply_rotary<KTraits>(
            (k_rope_offset == nullptr ? 0 : k_rope_offset[request_idx]) + chunk_start +
                iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq);
        sync_threads();
      }

      sync_threads();

      produce_v_r_(&v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, v_frag);

      lds_v<KTraits>(&v_smem, &v_smem_offset_r, v_r_frag);
      // compute attention score
      compute_qk<KTraits>(q_frag, k_r_frag, s_frag);

      produce_k_w<KTraits>(k_smem, &k_smem_offset_w, k_frag);

      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag, kv_head_idx);

      sync_threads();

      lds_k<KTraits>(&k_smem, &k_smem_offset_r, k_r_frag);

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      produce_k_r<KTraits>(&k_ptr, k_stride_n, (iter + 2) * CTA_TILE_KV, chunk_size, k_frag);

      // compute sfm*v
      compute_sfm_v_with_perm<KTraits>(s_frag, o_frag, d, v_r_frag);

      produce_v_w_(v_smem, &v_smem_offset_w, v_frag);
    }

#pragma unroll 1
    for (uint32_t iter = mask_iteration; iter < num_iterations; ++iter) {
      clear<DTypeQKAccum, NUM_MMA_Q * NUM_MMA_KV * 4>(s_frag[0][0]);

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        IdType* k_rope_offset = nullptr;
        if constexpr (has_maybe_k_rope_offset_v<Params>) {
          k_rope_offset = params.maybe_k_rope_offset;
        }
        k_smem_inplace_apply_rotary<KTraits>(
            (k_rope_offset == nullptr ? 0 : k_rope_offset[request_idx]) + chunk_start +
                iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq);
        sync_threads();
      }

      sync_threads();

      produce_v_r_(&v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, v_frag);

      lds_v<KTraits>(&v_smem, &v_smem_offset_r, v_r_frag);
      // compute attention score
      compute_qk<KTraits>(q_frag, k_r_frag, s_frag);

      produce_k_w<KTraits>(k_smem, &k_smem_offset_w, k_frag);

      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag, kv_head_idx);

      // apply mask
      logits_mask<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
          qo_len, kv_len, chunk_end, group_size, s_frag, kv_head_idx);

      sync_threads();

      lds_k<KTraits>(&k_smem, &k_smem_offset_r, k_r_frag);

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      produce_k_r<KTraits>(&k_ptr, k_stride_n, (iter + 2) * CTA_TILE_KV, chunk_size, k_frag);

      // compute sfm*v
      compute_sfm_v_with_perm<KTraits>(s_frag, o_frag, d, v_r_frag);

      produce_v_w_(v_smem, &v_smem_offset_w, v_frag);
    }
  } else {
#pragma unroll 1
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
      clear<DTypeQKAccum, NUM_MMA_Q * NUM_MMA_KV * 4>(s_frag[0][0]);
      sync_threads();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
        IdType* k_rope_offset = nullptr;
        if constexpr (has_maybe_k_rope_offset_v<Params>) {
          k_rope_offset = params.maybe_k_rope_offset;
        }
        k_smem_inplace_apply_rotary<KTraits>(
            (k_rope_offset == nullptr ? 0 : k_rope_offset[request_idx]) + chunk_start +
                iter * CTA_TILE_KV,
            &k_smem, &k_smem_offset_r, rope_freq);
        sync_threads();
      }
      produce_v_w_(v_smem, &v_smem_offset_w, v_frag);
      // compute attention score
      compute_qk<KTraits>(q_frag, &k_smem, &k_smem_offset_r, s_frag);
      produce_k_r<KTraits>(&k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, k_frag);
      logits_transform<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
          qo_len, kv_len, group_size, s_frag, kv_head_idx);

      // apply mask
      if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
        logits_mask<KTraits>(
            params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
            chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
            qo_len, kv_len, chunk_end, group_size, s_frag, kv_head_idx);
      }

      // compute m,d states in online softmax
      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

      sync_threads();
      // compute sfm*v
      compute_sfm_v_(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);
      produce_v_r_(&v_ptr, v_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, v_frag);
      produce_k_w<KTraits>(k_smem, &k_smem_offset_w, k_frag);
      sync_threads();
    }
  }

  sync_threads();
  finalize_m<KTraits>(variant, m);

  // normalize d
  normalize_d<KTraits>(o_frag, m, d);

  const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  // write back
  write_o_reg_gmem_(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                    /*o_stride_n=*/
                    partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                    /*o_stride_h=*/o_stride_h, group_size);

  // write lse
  if constexpr (AttentionVariant::use_softmax) {
    if (lse != nullptr) {
      if (get_warp_idx_kv<KTraits>() == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + lane_idx % 16 + mma_q * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_len) {
            if (partition_kv) {
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[mma_q]) + float(m[mma_q]);
            } else {
              lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                  math::ptx_log2(d[mma_q]) + float(m[mma_q]);
            }
          }
        }
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void batch_prefill_with_ragged_kv_cache_kernel_xc1000_ctk64(
    const Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q_64B = KTraits::UPCAST_STRIDE_Q_64B;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K_64B = KTraits::UPCAST_STRIDE_K_64B;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t K_THR_LAYOUT_ROW = KTraits::K_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t K_THR_LAYOUT_COL = KTraits::K_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr uint32_t V_THR_LAYOUT_ROW = KTraits::V_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t V_THR_LAYOUT_COL = KTraits::V_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

  using Selector = DeviceFunctionSelector<CTA_TILE_KV, false, KTraits>;
  constexpr auto write_o_reg_gmem_ = Selector::Write_O_Func;
  constexpr auto produce_v_w_ = Selector::Write_V_Func;
  constexpr auto produce_v_r_ = Selector::Read_V_Func;
  constexpr auto compute_sfm_v_ = Selector::Sfm_V_Func;

  DTypeQ* q = params.q;
  IdType* request_indices = params.request_indices;
  IdType* qo_tile_indices = params.qo_tile_indices;
  IdType* kv_tile_indices = params.kv_tile_indices;
  IdType* q_indptr = params.q_indptr;
  IdType* kv_indptr = params.kv_indptr;
  DTypeKV* k = params.k;
  DTypeKV* v = params.v;
  IdType* o_indptr = params.o_indptr;
  DTypeO* o = params.o;
  float* lse = params.lse;
  bool* block_valid_mask = params.block_valid_mask;
  const bool partition_kv = params.partition_kv;
  const uint32_t q_stride_n = params.q_stride_n;
  const uint32_t q_stride_h = params.q_stride_h;
  const uint32_t k_stride_n = params.k_stride_n;
  const uint32_t k_stride_h = params.k_stride_h;
  const uint32_t v_stride_n = params.v_stride_n;
  const uint32_t v_stride_h = params.v_stride_h;
  const int32_t maybe_window_left = params.window_left;
  const uint_fastdiv& group_size = params.group_size;

  static_assert(sizeof(DTypeQ) == 2);
  static_assert(CTA_TILE_KV == 64);

  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

  const uint32_t lane_idx = threadIdx.x, warp_idx = get_warp_idx<KTraits>();
  uint32_t bx, num_kv_heads, kv_head_idx;
  if constexpr (NUM_MMA_D_QK > NUM_MMA_D_VO) {
    bx = gridDim.z - blockIdx.z - 1;
    num_kv_heads = gridDim.x;
    kv_head_idx = blockIdx.x;
  } else {
    bx = blockIdx.x;
    num_kv_heads = gridDim.z;
    kv_head_idx = blockIdx.z;
  }

  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }

  const uint32_t num_qo_heads = group_size * num_kv_heads;
  const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
  const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                 window_left = variant.window_left;
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t qo_upper_bound = min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

  uint32_t q_frag[NUM_MMA_Q][NUM_MMA_D_QK][2];
  DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][4];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][4];
  DTypeQKAccum m[NUM_MMA_Q];
  float d[NUM_MMA_Q];
  float rope_freq[NUM_MMA_D_QK / 2][4];
  uint32_t k_frag[NUM_MMA_KV * 4 / NUM_WARPS_Q][NUM_MMA_D_QK / 4][2];

  init_states<KTraits>(variant, o_frag, m, d);

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>()) * NUM_MMA_Q * 16;
  smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;

  DTypeQ* q_ptr_base =
      q + q_indptr[request_idx] * q_stride_n + kv_head_idx * group_size * q_stride_h;

  DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h
                                    : o + o_indptr[request_idx] * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h;

  uint32_t q_smem_offset_r[4];
#pragma unroll
  for (uint32_t i = 0; i < 4; i++) {
    q_smem_offset_r[i] = qo_smem.template get_permuted_offset_64b<UPCAST_STRIDE_Q_64B>(
        get_warp_idx_q<KTraits>() * NUM_MMA_Q * 16 + lane_idx % 16, 4 * i + lane_idx / 16);
  }

  load_q_global_smem_64b<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                                  q_stride_h, group_size, &qo_smem);

  const uint32_t num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size, chunk_start))
           : chunk_size),
      CTA_TILE_KV);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                      qo_len + window_left + chunk_start),
               CTA_TILE_KV);

  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                        chunk_start))
           : chunk_size) /
      CTA_TILE_KV;

  smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);

  uint32_t k_smem_offset_w = k_smem.template get_permuted_offset_64b<UPCAST_STRIDE_K_64B>(
      warp_idx * 4 + lane_idx / 16, lane_idx % 16);
  DTypeKV* k_ptr =
      k + (kv_indptr[request_idx] + chunk_start + warp_idx * 4 + lane_idx / 16) * k_stride_n +
      kv_head_idx * k_stride_h + (lane_idx % 16) * upcast_size_64b<DTypeKV>();

  uint64_t* k_smem_w[NUM_MMA_KV * 4 / NUM_WARPS_Q][NUM_MMA_D_QK / 4];
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * 4 / NUM_WARPS_Q; ++i) {
#pragma unroll
    for (uint32_t j = 0; j < NUM_MMA_D_QK / 4; ++j) {
      k_smem_w[i][j] = (uint64_t*)k_smem.base +
                       k_smem.template get_permuted_offset_64b<UPCAST_STRIDE_K_64B>(
                           warp_idx * 4 + lane_idx / 16, lane_idx % 16) +
                       i * NUM_WARPS_Q * 4 * UPCAST_STRIDE_K_64B + j * 16;
    }
  }

  DTypeKV* v_ptr;
  uint32_t warpgroup_idx = warp_idx / 4;
  uint32_t warp_idx_in_wg = warp_idx % 4;
  if constexpr (NUM_MMA_KV % NUM_WARPS_Q == 0) {  // 4 waves
    v_ptr = v +
            (kv_indptr[request_idx] + chunk_start + warp_idx * V_THR_LAYOUT_ROW * 4 +
             lane_idx / V_THR_LAYOUT_COL * 4) *
                v_stride_n +
            kv_head_idx * v_stride_h + (lane_idx % V_THR_LAYOUT_COL) * upcast_size_64b<DTypeKV>();
  } else {  // 8 waves
    v_ptr = v +
            (kv_indptr[request_idx] + chunk_start + warp_idx_in_wg * V_THR_LAYOUT_ROW * 4 +
             lane_idx / V_THR_LAYOUT_COL * 4) *
                v_stride_n +
            kv_head_idx * v_stride_h +
            (lane_idx % V_THR_LAYOUT_COL + warpgroup_idx * V_THR_LAYOUT_COL) *
                upcast_size_64b<DTypeKV>();
  }

  auto& v_smem_w = [NUM_MMA_KV, NUM_WARPS_Q, NUM_MMA_D_VO]() -> auto& {
    if constexpr (NUM_MMA_KV % NUM_WARPS_Q == 0) {
      uint64_t* arr[NUM_MMA_D_VO / 4][4] = {};
      return arr;
    } else {
      uint64_t* arr[NUM_MMA_D_VO / 8][4] = {};
      return arr;
    }
  }();
  if constexpr (NUM_MMA_KV % NUM_WARPS_Q == 0) {  // 4 waves
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_D_VO / 4; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        v_smem_w[i][j] = (uint64_t*)v_smem.base +
                         v_smem.template get_64bx4_offset<UPCAST_STRIDE_V_64B>(
                             warp_idx * V_THR_LAYOUT_ROW + lane_idx / V_THR_LAYOUT_COL,
                             lane_idx % V_THR_LAYOUT_COL) +
                         i * 64 + j * 16;
      }
    }
  } else {  // 8 waves
#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_D_VO / 8; ++i) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        v_smem_w[i][j] = (uint64_t*)v_smem.base +
                         v_smem.template get_64bx4_offset<UPCAST_STRIDE_V_64B>(
                             warp_idx_in_wg * V_THR_LAYOUT_ROW + lane_idx / V_THR_LAYOUT_COL,
                             lane_idx % V_THR_LAYOUT_COL) +
                         warpgroup_idx * 64 + i * 128 + j * 16;
      }
    }
  }

  auto& v_frag = [NUM_MMA_KV, NUM_WARPS_Q, NUM_MMA_D_VO]() -> auto& {
    if constexpr (NUM_MMA_KV % NUM_WARPS_Q == 0) {
      uint32_t arr[NUM_MMA_KV / NUM_WARPS_Q * NUM_MMA_D_VO / (8 / sizeof(DTypeKV)) * 4 * 2] =
          {};  // arr[NUM_MMA_KV / NUM_WARPS_Q][NUM_MMA_D_VO / 4][4][2]
      return arr;
    } else {
      uint32_t arr[NUM_MMA_D_VO] = {};  // arr[1][NUM_MMA_D_VO / 8)][4][2]
      return arr;
    }
  }();

  uint64_t* k_smem_ptr_r[NUM_MMA_D_QK / 4][4][NUM_MMA_KV];
  uint64_t* v_smem_ptr_r[NUM_MMA_KV][NUM_MMA_D_VO / 4][4];
  calculate_smem_ptr_r<KTraits>(&k_smem, k_smem_ptr_r, &v_smem, v_smem_ptr_r);

  produce_k_r_64b<KTraits>(&k_ptr, k_stride_n, 0, chunk_size, k_frag);

  sync_threads();

  load_q_smem_reg_64b<KTraits>(&qo_smem, q_smem_offset_r, q_frag);

  sync_threads();

#pragma unroll 1
  for (uint32_t iter = 0; iter < mask_iteration; ++iter) {
    clear<DTypeQKAccum, NUM_MMA_Q * NUM_MMA_KV * 4>(s_frag[0][0]);

    produce_k_w_64b<KTraits>(k_smem_w, k_frag);

    produce_v_r_(&v_ptr, v_stride_n, iter * CTA_TILE_KV, chunk_size, v_frag);

    sync_threads();

    // compute attention score
    compute_qk<KTraits>(q_frag, k_smem_ptr_r, s_frag);

    produce_v_w_(v_smem_w, v_frag);

    produce_k_r_64b<KTraits>(&k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, k_frag);

    sync_threads();

    logits_transform<KTraits>(
        params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16, qo_len,
        kv_len, group_size, s_frag, kv_head_idx);

    // compute m,d states in online softmax
    update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

    // compute sfm*v
    compute_sfm_v_(v_smem_ptr_r, s_frag, o_frag, d);
  }

#pragma unroll 1
  for (uint32_t iter = mask_iteration; iter < num_iterations; ++iter) {
    clear<DTypeQKAccum, NUM_MMA_Q * NUM_MMA_KV * 4>(s_frag[0][0]);

    produce_k_w_64b<KTraits>(k_smem_w, k_frag);

    produce_v_r_(&v_ptr, v_stride_n, iter * CTA_TILE_KV, chunk_size, v_frag);

    sync_threads();

    // compute attention score
    compute_qk<KTraits>(q_frag, k_smem_ptr_r, s_frag);

    produce_v_w_(v_smem_w, v_frag);

    produce_k_r_64b<KTraits>(&k_ptr, k_stride_n, (iter + 1) * CTA_TILE_KV, chunk_size, k_frag);

    sync_threads();

    logits_transform<KTraits>(
        params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16, qo_len,
        kv_len, group_size, s_frag, kv_head_idx);

    // apply mask
    logits_mask<KTraits>(
        params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16, qo_len,
        kv_len, chunk_end, group_size, s_frag, kv_head_idx);

    // compute m,d states in online softmax
    update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

    // compute sfm*v
    compute_sfm_v_(v_smem_ptr_r, s_frag, o_frag, d);
  }

  sync_threads();
  finalize_m<KTraits>(variant, m);

  // normalize d
  normalize_d<KTraits>(o_frag, m, d);

  const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  // write back
  write_o_reg_gmem_(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                    /*o_stride_n=*/
                    partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                    /*o_stride_h=*/o_stride_h, group_size);

  // write lse
  if constexpr (AttentionVariant::use_softmax) {
    if (lse != nullptr) {
      if (get_warp_idx_kv<KTraits>() == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + lane_idx % 16 + mma_q * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_len) {
            if (partition_kv) {
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[mma_q]) + float(m[mma_q]);
            } else {
              lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                  math::ptx_log2(d[mma_q]) + float(m[mma_q]);
            }
          }
        }
      }
    }
  }
}

template <typename KTraits, typename Params>
__device__ __forceinline__ void batch_prefill_with_paged_kv_cache_kernel_xc1000(
    const Params params) {
  using DTypeQ = typename Params::DTypeQ;
  using DTypeKV = typename Params::DTypeKV;
  using DTypeO = typename Params::DTypeO;
  using IdType = typename Params::IdType;
  using DTypeQKAccum = typename KTraits::DTypeQKAccum;
  using AttentionVariant = typename KTraits::AttentionVariant;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_Q = KTraits::NUM_MMA_Q;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_KV = KTraits::NUM_MMA_KV;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_QK = KTraits::NUM_MMA_D_QK;
  [[maybe_unused]] constexpr uint32_t NUM_MMA_D_VO = KTraits::NUM_MMA_D_VO;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_QK = KTraits::HEAD_DIM_QK;
  [[maybe_unused]] constexpr uint32_t HEAD_DIM_VO = KTraits::HEAD_DIM_VO;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_K = KTraits::UPCAST_STRIDE_K;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_V_64B = KTraits::UPCAST_STRIDE_V_64B;
  [[maybe_unused]] constexpr uint32_t UPCAST_STRIDE_O = KTraits::UPCAST_STRIDE_O;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_Q = KTraits::NUM_WARPS_Q;
  [[maybe_unused]] constexpr uint32_t NUM_WARPS_KV = KTraits::NUM_WARPS_KV;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_Q = KTraits::SWIZZLE_MODE_Q;
  [[maybe_unused]] constexpr SwizzleMode SWIZZLE_MODE_KV = KTraits::SWIZZLE_MODE_KV;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_Q = KTraits::CTA_TILE_Q;
  [[maybe_unused]] constexpr uint32_t CTA_TILE_KV = KTraits::CTA_TILE_KV;
  [[maybe_unused]] constexpr uint32_t K_THR_LAYOUT_ROW = KTraits::K_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t K_THR_LAYOUT_COL = KTraits::K_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr uint32_t V_THR_LAYOUT_ROW = KTraits::V_THR_LAYOUT_ROW;
  [[maybe_unused]] constexpr uint32_t V_THR_LAYOUT_COL = KTraits::V_THR_LAYOUT_COL;
  [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;

  IdType* request_indices = params.request_indices;
  IdType* qo_tile_indices = params.qo_tile_indices;
  IdType* kv_tile_indices = params.kv_tile_indices;
  DTypeQ* q = params.q;
  IdType* q_indptr = params.q_indptr;
  IdType* o_indptr = params.o_indptr;
  DTypeO* o = params.o;
  float* lse = params.lse;
  bool* block_valid_mask = params.block_valid_mask;
  const paged_kv_t<DTypeKV, IdType>& paged_kv = params.paged_kv;
  const bool partition_kv = params.partition_kv;
  const int32_t maybe_window_left = params.window_left;
  const uint_fastdiv& group_size = params.group_size;

  static_assert(sizeof(DTypeQ) == 2);
  const uint32_t kv_chunk_size = *(params.kv_chunk_size_ptr);

  const uint32_t bx = blockIdx.x, lane_idx = threadIdx.x, warp_idx = get_warp_idx<KTraits>(),
                 kv_head_idx = blockIdx.z;
  if (block_valid_mask && !block_valid_mask[bx]) {
    return;
  }
  const uint32_t num_kv_heads = gridDim.z, num_qo_heads = num_kv_heads * group_size;
  const uint32_t request_idx = request_indices[bx], qo_tile_idx = qo_tile_indices[bx],
                 kv_tile_idx = kv_tile_indices[bx];
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  AttentionVariant variant(params, /*batch_idx=*/request_idx, smem);
  const uint32_t qo_len = variant.qo_len, kv_len = variant.kv_len,
                 window_left = variant.window_left;
  const uint32_t kv_len_safe = kv_len > 0 ? kv_len : 1;
  const uint32_t max_chunk_size = partition_kv ? kv_chunk_size : kv_len;
  const uint32_t chunk_start = partition_kv ? kv_tile_idx * max_chunk_size : 0;
  const uint32_t chunk_end =
      partition_kv ? min((kv_tile_idx + 1) * max_chunk_size, kv_len) : kv_len;
  const uint32_t chunk_size = chunk_end - chunk_start;
  const uint32_t qo_upper_bound = min(qo_len, ceil_div((qo_tile_idx + 1) * CTA_TILE_Q, group_size));

  uint32_t q_frag[NUM_MMA_Q][NUM_MMA_D_QK / 2][4];
  DTypeQKAccum s_frag[NUM_MMA_Q][NUM_MMA_KV][4];
  alignas(16) float o_frag[NUM_MMA_Q][NUM_MMA_D_VO][4];
  DTypeQKAccum m[NUM_MMA_Q];
  float d[NUM_MMA_Q];
  float rope_freq[NUM_MMA_D_QK / 2][4];

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    const float rope_rcp_scale = params.rope_rcp_scale;
    const float rope_rcp_theta = params.rope_rcp_theta;
    init_rope_freq<KTraits>(rope_freq, rope_rcp_scale, rope_rcp_theta);
  }
  init_states<KTraits>(variant, o_frag, m, d);

  const uint32_t qo_packed_idx_base =
      (qo_tile_idx * NUM_WARPS_Q + get_warp_idx_q<KTraits>()) * NUM_MMA_Q * 16;
  const uint32_t q_stride_n = params.q_stride_n, q_stride_h = params.q_stride_h;
  smem_t<SWIZZLE_MODE_Q> qo_smem(smem_storage.q_smem);
  const uint32_t o_stride_n = num_qo_heads * HEAD_DIM_VO, o_stride_h = HEAD_DIM_VO;
  DTypeQ* q_ptr_base =
      q + q_indptr[request_idx] * q_stride_n + (kv_head_idx * group_size) * q_stride_h;
  DTypeO* o_ptr_base = partition_kv ? o + (o_indptr[request_idx] + kv_tile_idx) * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h
                                    : o + o_indptr[request_idx] * o_stride_n +
                                          (kv_head_idx * group_size) * o_stride_h;

  uint32_t q_smem_offset_r = qo_smem.template get_permuted_offset<UPCAST_STRIDE_Q>(
      get_warp_idx_q<KTraits>() * NUM_MMA_Q * 16 + lane_idx % 16, lane_idx / 16);

  load_q_global_smem<KTraits>(qo_packed_idx_base, qo_upper_bound, q_ptr_base, q_stride_n,
                              q_stride_h, group_size, &qo_smem);
  sync_threads();
  load_q_smem_reg<KTraits>(&qo_smem, &q_smem_offset_r, q_frag);

  if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
    sync_threads();
    IdType* q_rope_offset = nullptr;
    if constexpr (has_maybe_q_rope_offset_v<Params>) {
      q_rope_offset = params.maybe_q_rope_offset;
    }
    if (q_rope_offset == nullptr) {
      q_smem_inplace_apply_rotary<KTraits>(qo_packed_idx_base, qo_len, kv_len, group_size, &qo_smem,
                                           &q_smem_offset_r, rope_freq);
    } else {
      q_smem_inplace_apply_rotary_with_pos<KTraits>(qo_packed_idx_base,
                                                    q_rope_offset + q_indptr[request_idx], &qo_smem,
                                                    group_size, &q_smem_offset_r, rope_freq);
    }
    sync_threads();
  }

  smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
  size_t thr_local_k_offset[NUM_MMA_KV * K_THR_LAYOUT_COL / 4 / NUM_WARPS_Q];
  size_t thr_local_v_offset[NUM_MMA_KV / NUM_WARPS_Q / (V_THR_LAYOUT_ROW / 4)][4];

  uint32_t k_smem_offset_r = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + lane_idx % 16, lane_idx / 16),
           v_smem_offset_r = v_smem.template get_64bx4_offset<UPCAST_STRIDE_V_64B>(
               get_warp_idx_kv<KTraits>() * NUM_MMA_KV * 16 + lane_idx / 16, lane_idx % 16),
           k_smem_offset_w = k_smem.template get_permuted_offset<UPCAST_STRIDE_K>(
               warp_idx * K_THR_LAYOUT_ROW + lane_idx / K_THR_LAYOUT_COL,
               lane_idx % K_THR_LAYOUT_COL),
           v_smem_offset_w = v_smem.template get_64bx4_offset<UPCAST_STRIDE_V>(
               warp_idx * V_THR_LAYOUT_ROW + lane_idx / V_THR_LAYOUT_COL,
               lane_idx % V_THR_LAYOUT_COL * 2);
  const IdType last_indptr = paged_kv.indptr[paged_kv.batch_size];

  uint32_t packed_page_iter_base = paged_kv.indptr[request_idx] * paged_kv.page_size + chunk_start;
#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV * K_THR_LAYOUT_COL / 4 / NUM_WARPS_Q; ++i) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * K_THR_LAYOUT_ROW +
                                  lane_idx / K_THR_LAYOUT_COL +
                                  K_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                              page_iter, entry_idx);
    thr_local_k_offset[i] = paged_kv.protective_get_kv_offset(
        page_iter, kv_head_idx, entry_idx, (lane_idx % K_THR_LAYOUT_COL) * upcast_size<DTypeKV>(),
        last_indptr);
  }
  sync_threads();  // k shares smem with q
  page_produce_k<KTraits>(k_smem, &k_smem_offset_w, paged_kv, 0, thr_local_k_offset, chunk_size);

#pragma unroll
  for (uint32_t i = 0; i < NUM_MMA_KV / NUM_WARPS_Q / (V_THR_LAYOUT_ROW / 4); ++i) {
#pragma unroll
    for (uint32_t j = 0; j < 4; ++j) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * V_THR_LAYOUT_ROW * 4 +
                                    lane_idx / V_THR_LAYOUT_COL * 4 + j +
                                    V_THR_LAYOUT_ROW * 4 * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      thr_local_v_offset[i][j] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx,
          (lane_idx % V_THR_LAYOUT_COL) * upcast_size_64b<DTypeKV>(), last_indptr);
    }
  }
  page_produce_v<KTraits>(v_smem, &v_smem_offset_w, paged_kv, 0, thr_local_v_offset, chunk_size);

  const uint32_t num_iterations = ceil_div(
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(
                     kv_len - qo_len + ((qo_tile_idx + 1) * CTA_TILE_Q) / group_size, chunk_start))
           : chunk_size),
      CTA_TILE_KV);

  const uint32_t window_iteration =
      ceil_div(sub_if_greater_or_zero(kv_len + (qo_tile_idx + 1) * CTA_TILE_Q / group_size,
                                      qo_len + window_left + chunk_start),
               CTA_TILE_KV);

  const uint32_t mask_iteration =
      (MASK_MODE == MaskMode::kCausal
           ? min(chunk_size,
                 sub_if_greater_or_zero(kv_len + (qo_tile_idx * CTA_TILE_Q) / group_size - qo_len,
                                        chunk_start))
           : chunk_size) /
      CTA_TILE_KV;

#pragma unroll 1
  for (uint32_t iter = 0; iter < num_iterations; ++iter) {
    clear<DTypeQKAccum, NUM_MMA_Q * NUM_MMA_KV * 4>(s_frag[0][0]);
    sync_threads();
    packed_page_iter_base += CTA_TILE_KV;

#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV * K_THR_LAYOUT_COL / 4 / NUM_WARPS_Q; ++i) {
      uint32_t page_iter, entry_idx;
      paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * K_THR_LAYOUT_ROW +
                                    lane_idx / K_THR_LAYOUT_COL +
                                    K_THR_LAYOUT_ROW * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                page_iter, entry_idx);
      thr_local_k_offset[i] = paged_kv.protective_get_kv_offset(
          page_iter, kv_head_idx, entry_idx, (lane_idx % K_THR_LAYOUT_COL) * upcast_size<DTypeKV>(),
          last_indptr);
    }

#pragma unroll
    for (uint32_t i = 0; i < NUM_MMA_KV / NUM_WARPS_Q / (V_THR_LAYOUT_ROW / 4); ++i) {
#pragma unroll
      for (uint32_t j = 0; j < 4; ++j) {
        uint32_t page_iter, entry_idx;
        paged_kv.page_size.divmod(packed_page_iter_base + warp_idx * V_THR_LAYOUT_ROW * 4 +
                                      lane_idx / V_THR_LAYOUT_COL * 4 + j +
                                      V_THR_LAYOUT_ROW * 4 * NUM_WARPS_Q * NUM_WARPS_KV * i,
                                  page_iter, entry_idx);
        thr_local_v_offset[i][j] = paged_kv.protective_get_kv_offset(
            page_iter, kv_head_idx, entry_idx,
            (lane_idx % V_THR_LAYOUT_COL) * upcast_size_64b<DTypeKV>(), last_indptr);
      }
    }

    sync_threads();

    if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {
      k_smem_inplace_apply_rotary<KTraits>(
          (paged_kv.rope_pos_offset == nullptr ? 0 : paged_kv.rope_pos_offset[request_idx]) +
              chunk_start + iter * CTA_TILE_KV,
          &k_smem, &k_smem_offset_r, rope_freq);
      sync_threads();
    }

    // compute attention score
    compute_qk<KTraits>(q_frag, &k_smem, &k_smem_offset_r, s_frag);

    logits_transform<KTraits>(
        params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
        chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16, qo_len,
        kv_len, group_size, s_frag, kv_head_idx);

    // apply mask
    if (MASK_MODE == MaskMode::kCustom || (iter >= mask_iteration || iter < window_iteration)) {
      logits_mask<KTraits>(
          params, variant, /*batch_idx=*/request_idx, qo_packed_idx_base,
          chunk_start + (iter * NUM_WARPS_KV + get_warp_idx_kv<KTraits>()) * NUM_MMA_KV * 16,
          qo_len, kv_len, chunk_end, group_size, s_frag, kv_head_idx);
    }

    // compute m,d states in online softmax
    update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);

    sync_threads();
    page_produce_k<KTraits>(k_smem, &k_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                            thr_local_k_offset, chunk_size);
    sync_threads();

    // compute sfm*v
    compute_sfm_v<KTraits>(&v_smem, &v_smem_offset_r, s_frag, o_frag, d);

    sync_threads();
    page_produce_v<KTraits>(v_smem, &v_smem_offset_w, paged_kv, (iter + 1) * CTA_TILE_KV,
                            thr_local_v_offset, chunk_size);
  }
  sync_threads();

  finalize_m<KTraits>(variant, m);

  // normalize d
  normalize_d<KTraits>(o_frag, m, d);

  const uint32_t num_kv_chunks = (kv_len_safe + kv_chunk_size - 1) / kv_chunk_size;

  // write_back
  write_o_reg_gmem<KTraits>(o_frag, &qo_smem, o_ptr_base, qo_packed_idx_base, qo_len,
                            /*o_stride_n=*/
                            partition_kv ? num_kv_chunks * o_stride_n : o_stride_n,
                            /*o_stride_h=*/o_stride_h, group_size);

  // write lse
  if constexpr (variant.use_softmax) {
    if (lse != nullptr) {
      if (get_warp_idx_kv<KTraits>() == 0) {
#pragma unroll
        for (uint32_t mma_q = 0; mma_q < NUM_MMA_Q; ++mma_q) {
          uint32_t q, r;
          group_size.divmod(qo_packed_idx_base + lane_idx % 16 + mma_q * 16, q, r);
          const uint32_t qo_head_idx = kv_head_idx * group_size + r;
          const uint32_t qo_idx = q;
          if (qo_idx < qo_upper_bound) {
            if (partition_kv) {
              lse[(o_indptr[request_idx] + qo_idx * num_kv_chunks + kv_tile_idx) * num_qo_heads +
                  qo_head_idx] = math::ptx_log2(d[mma_q]) + float(m[mma_q]);
            } else {
              lse[(o_indptr[request_idx] + qo_idx) * num_qo_heads + qo_head_idx] =
                  math::ptx_log2(d[mma_q]) + float(m[mma_q]);
            }
          }
        }
      }
    }
  }
}

}  // namespace flashinfer

#endif  // FLASHINFER_PREFILL_KERNELS_XCORE1000_CUH_
