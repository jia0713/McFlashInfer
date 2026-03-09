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
#ifndef FLASHINFER_PERMUTED_SMEM_CUH_
#define FLASHINFER_PERMUTED_SMEM_CUH_

#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include <cuda/pipeline>

#include "cp_async.cuh"
#include "mma.cuh"

namespace flashinfer {

enum class SwizzleMode {
  k64B,
  k128B,
};

// Use 128bit as the granularity to fetch/store data per thread to maximize memory bandwidth
using b128_t = uint4;

/*!
 * \brief Compute the number of elements that can be stored in a b128_t.
 * \tparam T The data type of the elements.
 */
template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t upcast_size() {
  return sizeof(b128_t) / sizeof(T);
}

template <typename T>
constexpr __host__ __device__ __forceinline__ uint32_t upcast_size_64b() {
  return sizeof(uint64_t) / sizeof(T);
}

/*!
 * \brief The shared memory wrapper.
 */
template <SwizzleMode swizzle_mode>
struct smem_t {
  // The base pointer.
  b128_t* base;
  __device__ __forceinline__ smem_t() : base(nullptr) {}
  template <typename T>
  __device__ __forceinline__ smem_t(T* base) : base((b128_t*)base) {}

  /*!
   * \brief Compute the element offset given coordinates in a permuted shared memory.
   * \tparam stride The stride (in terms of b128_t's) in the permuted shared memory.
   * \tparam rows The max row of swizzle block, 8 for b128 and 16 for b64.
   * \param i The row index.
   * \param j The column index.
   */
  template <uint32_t stride, uint32_t rows = 8>
  static __device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      if constexpr (rows == 4) {
        // sts for lds_trans_4x16_b64
        return i * stride + (j ^ (i % rows)) * 2;
      } else {
        return i * stride + (j ^ (i % rows));
      }
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(stride == 4);
      return i * stride + (j ^ ((i / 2) % 4));
    }
  }

  template <uint32_t stride, uint32_t rows = 16>
  static __device__ __forceinline__ uint32_t get_permuted_offset_64b(uint32_t i, uint32_t j) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      if constexpr (rows == 4) {
        // lds for lds_trans_4x16_b64
        return i * stride + (j ^ (i % rows)) * 4;
      } else if constexpr (rows == 8) {
        // used for ldg_b128
        return i * stride + (j ^ (i % rows)) * 2;
      } else if constexpr (rows == 16) {
        return i * stride + (j ^ (i % rows));
      } else {
        FLASHINFER_RUNTIME_ASSERT("not support");
      }
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(stride == 8);
      return i * stride + (j ^ ((i / 2) % 8));
    }
  }

  template <uint32_t stride>
  static __device__ __forceinline__ uint32_t get_64bx4_offset(uint32_t i, uint32_t j) {
    static_assert(swizzle_mode == SwizzleMode::k128B);
    return i * stride * 4 + j;
  }

  // get the offset in the swizzle block(8x64_f16_128b)
  // offset = swz_block_x * 64 + swz_block_y * 8 * UPCAST_STRIDE
  template <bool enable_lds_trans = false>
  static __device__ __forceinline__ uint32_t get_swizzle_offset(uint32_t offset, uint32_t i,
                                                                uint32_t j) {
    static_assert(swizzle_mode == SwizzleMode::k128B);
    if constexpr (enable_lds_trans) {
      return offset + i * 8 + (j ^ (i % 4)) * 2;
    } else {
      return offset + i * 8 + j ^ i;
    }
  }

  // get the offset in the swizzle block(8x64_f16_64b)
  template <bool enable_lds_trans = false>
  static __device__ __forceinline__ uint32_t get_swizzle_offset_64b(uint32_t offset, uint32_t i,
                                                                    uint32_t j) {
    static_assert(swizzle_mode == SwizzleMode::k128B);
    if constexpr (enable_lds_trans) {
      return offset + i * 16 + (j ^ (i % 4)) * 4;
    } else {
      return offset + i * 16 + j ^ i;
    }
  }

  template <uint32_t step_size>
  static __device__ __forceinline__ uint32_t advance_offset_by_column(uint32_t offset,
                                                                      uint32_t step_idx = 0) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      static_assert(step_size == 2 || step_size == 4 || step_size % 8 == 0,
                    "Unsupported step size");
      if constexpr (step_size == 2) {
        return (offset ^ (0x2 + (0x4 * (step_idx % 2 == 1)))) + (step_idx % 4 == 3) * 8;
      } else if constexpr (step_size == 4) {
        return (offset ^ 0x4) + (step_idx % 2 == 1) * 8;
      } else {
        // step_size % 8 == 0
        return offset + step_size;
      }
    } else {
      // swizzle_mode == SwizzleMode::k64B
      static_assert(step_size == 2, "Unsupported step size");
      return (offset ^ 0x2) + (step_idx % 2 == 1) * 4;
    }
  }

  template <uint32_t step_size, uint32_t row_stride>
  static __device__ __forceinline__ uint32_t advance_offset_by_row(uint32_t offset) {
    if constexpr (swizzle_mode == SwizzleMode::k128B) {
      return offset + step_size * row_stride;
    } else {
      static_assert(step_size == 4 || step_size % 8 == 0, "Unsupported step size");
      if constexpr (step_size == 4) {
        return (offset ^ 0x2) + step_size * row_stride;
      } else {
        // step_size % 8 == 0
        return offset + step_size * row_stride;
      }
    }
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_left_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_left_half(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_right_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_right_half(R, smem_ptr);
  }

  __device__ __forceinline__ void stmatrix_m8n8x4(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::stmatrix_m8n8x4(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_left_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans_left_half(R, smem_ptr);
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_right_half(uint32_t offset, uint32_t* R) {
    b128_t* smem_ptr = base + offset;
    mma::ldmatrix_m8n8x4_trans_right_half(R, smem_ptr);
  }

  template <cp_async::SharedMemFillMode fill_mode, typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr, bool predicate) {
    b128_t* smem_ptr = base + offset;
    cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch, fill_mode>(
        smem_ptr, reinterpret_cast<const b128_t*>(gptr), predicate);
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr) {
    b128_t* smem_ptr = base + offset;
    cp_async::load_128b<cp_async::PrefetchMode::kPrefetch>(smem_ptr,
                                                           reinterpret_cast<const b128_t*>(gptr));
  }

  template <typename T>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr, bool predicate) {
    b128_t* smem_ptr = base + offset;
    cp_async::load_128b_bsm_pred(reinterpret_cast<T*>(smem_ptr), gptr, predicate);
  }

  template <typename T, bool Is_even_MN = false>
  __device__ __forceinline__ void load_128b_async(uint32_t offset, const T* gptr,
                                                  bool predicate = 1) {
    b128_t* smem_ptr = base + offset;
    if constexpr (Is_even_MN) {
      cp_async::load_128b_bsm(reinterpret_cast<T*>(smem_ptr), gptr);
    } else {
      cp_async::load_128b_bsm_pred(reinterpret_cast<T*>(smem_ptr), gptr, predicate);
    }
  }

  __device__ __forceinline__ void load_128b(uint32_t offset, uint32_t* frag) {
    b128_t* smem_ptr = base + offset;
    *(b128_t*)frag = *smem_ptr;
  }

  __device__ __forceinline__ void load_64b(uint32_t offset, uint32_t* frag) {
    uint64_t* smem_ptr = (uint64_t*)base + offset;
    *(uint64_t*)frag = *smem_ptr;
  }

  __device__ __forceinline__ void load_32b(uint32_t offset, void* frag) {
    uint32_t* smem_ptr = (uint32_t*)base + offset;
    *(uint32_t*)frag = *smem_ptr;
  }

  __device__ __forceinline__ void load_16b(uint32_t offset, void* frag) {
    uint16_t* smem_ptr = (uint16_t*)base + offset;
    *(uint16_t*)frag = *smem_ptr;
  }

  __device__ __forceinline__ void store_128b(uint32_t offset, uint32_t* frag) {
    b128_t* smem_ptr = base + offset;
    *smem_ptr = *(b128_t*)frag;
  }

  template <typename T>
  __device__ __forceinline__ void store_global_128b(uint32_t offset, T* gptr) {
    *reinterpret_cast<b128_t*>(gptr) = *(base + offset);
  }

  __device__ __forceinline__ void store_64b(uint32_t offset, uint32_t* frag) {
    uint64_t* smem_ptr = (uint64_t*)base + offset;
    *smem_ptr = *(uint64_t*)frag;
  }

  __device__ __forceinline__ void load_64b_trans(uint32_t offset, uint32_t* frag) {
    uint64_t* smem_ptr = (uint64_t*)base + offset;
    *(uint64_t*)frag = __builtin_mxc_load_shared_trans_4x16_i64((int64_t*)smem_ptr);
  }
};

__device__ __forceinline__ void smem_load_64b(uint64_t* smem_ptr, uint32_t* frag) {
  *(uint64_t*)frag = *smem_ptr;
}

__device__ __forceinline__ void smem_store_64b(uint64_t* smem_ptr, uint32_t* frag) {
  *smem_ptr = *(uint64_t*)frag;
}

}  // namespace flashinfer

#endif  // FLASHINFER_PERMUTED_SMEM_CUH_
