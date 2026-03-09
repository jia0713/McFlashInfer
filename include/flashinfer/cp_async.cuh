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
#ifndef FLASHINFER_CP_ASYNC_CUH_
#define FLASHINFER_CP_ASYNC_CUH_

#include <mc_runtime.h>

#include <cstdint>

namespace flashinfer {

namespace cp_async {

enum class SharedMemFillMode {
  kFillZero,  // Fill zero to shared memory when predicate is false
  kNoFill     // Do not fill zero to shared memory when predicate is false
};

enum class PrefetchMode {
  kNoPrefetch,  // Do not fetch additional data from global memory to L2
  kPrefetch     // Fetch additional data from global memory to L2
};

/*!
 * \brief Wrapper of PTX cp.async.commit_group instruction, commit all prior uncommitted
 *   cp.async instructions to a group
 */
__device__ __forceinline__ void commit_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.wait_group instruction
 * \tparam n Wait till most recent n groups are committed
 */
template <size_t n>
__device__ __forceinline__ void wait_group() {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load_128b(T* smem_ptr, const T* gmem_ptr) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                 "l"(gmem_ptr), "n"(16), "r"(16));
  }
#else
  *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
#endif
}

/*!
 * \brief Wrapper of PTX cp.async.cg.shared.global instruction, asynchronously copy data from
 *   global memory to shared memory with predicate.
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load_128b(T* smem_ptr, const T* gmem_ptr, bool predicate) {
#ifdef FLASHINFER_CP_ASYNC_ENABLED
  uint32_t smem_int_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
    int src_in_bytes = predicate ? 16 : 0;
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    } else {
      asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n" ::"r"(smem_int_ptr),
                   "l"(gmem_ptr), "n"(16), "r"(src_in_bytes));
    }
  } else {
    if constexpr (prefetch_mode == PrefetchMode::kPrefetch) {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global.L2::128B [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    } else {
      asm volatile(
          "{\n"
          " .reg .pred p;\n"
          " setp.ne.b32 p, %0, 0;\n"
          " @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)predicate),
          "r"(smem_int_ptr), "l"(gmem_ptr), "n"(16));
    }
  }
#else
  if (predicate) {
    *((uint4*)smem_ptr) = *((uint4*)gmem_ptr);
  } else {
    if constexpr (fill_mode == SharedMemFillMode::kFillZero) {
      *((uint4*)smem_ptr) = make_uint4(0, 0, 0, 0);
    }
  }
#endif
}

/*!
 * \brief Load specified number of bits per thread from global memory to shared memory
 * \tparam num_bits Number of bits to load, must be 128 or 256
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 */
template <size_t num_bits, PrefetchMode prefetch_mode, typename T>
__device__ __forceinline__ void load(T* smem_ptr, const T* gmem_ptr) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
  } else {
    load_128b<prefetch_mode>(smem_ptr, gmem_ptr);
    load_128b<prefetch_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T));
  }
}

/*!
 * \brief Load specified number of bits per thread from global memory to shared memory with
 *   predicate
 * \tparam num_bits Number of bits to load, must be 128 or 256
 * \tparam prefetch_mode Whether to fetch additional data from global memory to L2
 * \tparam fill_mode Whether to fill zero to shared memory when predicate is false
 * \tparam T Data type
 * \param smem_ptr Pointer to shared memory
 * \param gmem_ptr Pointer to global memory
 * \param predicate Predicate value
 * \note fill zero is slower than not fill zero
 */
template <size_t num_bits, PrefetchMode prefetch_mode, SharedMemFillMode fill_mode, typename T>
__device__ __forceinline__ void pred_load(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  static_assert(num_bits == 128 || num_bits == 256, "num_bits must be 128 or 256");
  if constexpr (num_bits == 128) {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
  } else {
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr, gmem_ptr, predicate);
    pred_load_128b<prefetch_mode, fill_mode>(smem_ptr + 16 / sizeof(T), gmem_ptr + 16 / sizeof(T),
                                             predicate);
  }
}

template <typename T>
__device__ __forceinline__ void load_128b_pred(uint32_t* frag, const T* gmem_ptr, bool predicate) {
  typedef __NATIVE_VECTOR__(4, int) VecType;
  auto src_ptr = (VecType*)gmem_ptr;
  auto dst_ptr = (VecType*)frag;
  *dst_ptr = __builtin_mxc_ldg_b128_predicator(src_ptr, 0, true, true, false, false, predicate, 1,
                                               MACA_ICMP_EQ);
}

template <typename T>
__device__ __forceinline__ void load_32b_pred(uint32_t* frag, const T* gmem_ptr, bool predicate) {
  typedef __NATIVE_VECTOR__(1, int) VecType;
  auto src_ptr = (VecType*)gmem_ptr;
  auto dst_ptr = (VecType*)frag;
  *dst_ptr = __builtin_mxc_ldg_b32_predicator(src_ptr, 0, true, true, false, false, predicate, 1,
                                              MACA_ICMP_EQ);
}

template <typename T>
__device__ __forceinline__ void load_128b_bsm_pred(T* smem_ptr, const T* gmem_ptr, bool predicate) {
  typedef __NATIVE_VECTOR__(4, int) VecType;
  auto src_ptr = (VecType*)gmem_ptr;
  auto dst_ptr = (VecType*)smem_ptr;
  __builtin_mxc_ldg_b128_bsm_predicator(dst_ptr, src_ptr, 0, true, true, false, true, predicate, 1,
                                        MACA_ICMP_EQ);
}

template <typename T>
__device__ __forceinline__ void load_128b_bsm(T* smem_ptr, const T* gmem_ptr) {
  typedef __NATIVE_VECTOR__(4, int) VecType;
  auto src_ptr = (VecType*)gmem_ptr;
  auto dst_ptr = (VecType*)smem_ptr;
  __builtin_mxc_ldg_b128_bsm(dst_ptr, src_ptr, 0, -1, true, true, false, true);
}

template <typename T>
__device__ __forceinline__ void load_64b_pred(uint32_t* frag, const T* gmem_ptr, bool predicate) {
  typedef __NATIVE_VECTOR__(2, int) VecType;
  auto src_ptr = (VecType*)gmem_ptr;
  auto dst_ptr = (VecType*)frag;
  *dst_ptr = __builtin_mxc_ldg_b64_predicator(src_ptr, 0, true, true, false, false, predicate, 1,
                                              MACA_ICMP_EQ);
}

template <typename T>
__device__ __forceinline__ void store_64b_pred(uint32_t* frag, T* gmem_ptr, bool predicate) {
  auto src_ptr = (uint64_t*)frag;
  auto dst_ptr = (uint64_t*)gmem_ptr;
  __builtin_mxc_stg_b64_predicator(dst_ptr, 0, *src_ptr, true, false, false, predicate, 1,
                                   MACA_ICMP_EQ);
}

template <typename T>
__device__ __forceinline__ void store_128b_pred(uint32_t* frag, T* gmem_ptr, bool predicate) {
  typedef __NATIVE_VECTOR__(4, int) VecType;
  auto src_ptr = (VecType*)frag;
  auto dst_ptr = (VecType*)gmem_ptr;
  __builtin_mxc_stg_b128_predicator(dst_ptr, 0, *src_ptr, true, false, false, predicate, 1,
                                    MACA_ICMP_EQ);
}

// get gmem swizzle offset
template <uint32_t row = 8>
__device__ __forceinline__ uint32_t get_permuted_offset(uint32_t i, uint32_t j) {
  if constexpr (row == 4) {
    // for 256b element(used for lds_trans), we need to multiply by 2 to get the correct offset
    // because the max load bitwidth is 128b
    return (j ^ (i % 4)) * 2;
  } else {
    return j ^ (i % row);
  }
}

// This function only can be used in the loop unrolling scene.
// fill_mode: Whether to fill zero to shared memory when predicate is false,
// true: fill zero, false: not fill zero
template <bool fill_mode = false, typename T>
__device__ __forceinline__ b128vectype pred_load_128b(T* smem_ptr, const T* gmem_ptr,
                                                      bool predicate) {
  return memcpy_async_pred<16, MACA_ICMP_EQ, fill_mode>(
      reinterpret_cast<b128vectype*>(smem_ptr),
      reinterpret_cast<b128vectype*>(const_cast<T*>(gmem_ptr)), predicate, true);
}

}  // namespace cp_async

}  // namespace flashinfer

#endif  // FLASHINFER_CP_ASYNC_CUH_
