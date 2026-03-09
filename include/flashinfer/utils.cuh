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
#ifndef FLASHINFER_UTILS_CUH_
#define FLASHINFER_UTILS_CUH_
#include <cuda_device_runtime_api.h>
#include <cuda_fp8.h>
#include <maca_bfloat16.h>
#include <maca_fp16.h>
#include <mc_runtime.h>

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include "exception.h"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// macro to turn off fp16 qk reduction to reduce binary
#ifndef FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION
#define FLASHINFER_ALWAYS_DISUSE_FP16_QK_REDUCTION 0
#endif

#ifndef NDEBUG
#define FLASHINFER_CUDA_CALL(func, ...)                                                     \
  {                                                                                         \
    cudaError_t e = (func);                                                                 \
    if (e != cudaSuccess) {                                                                 \
      std::cerr << "CUDA Error: " << cudaGetErrorString(e) << " (" << e << ") " << __FILE__ \
                << ": line " << __LINE__ << " at function " << STR(func) << std::endl;      \
      return e;                                                                             \
    }                                                                                       \
  }
#else
#define FLASHINFER_CUDA_CALL(func, ...) \
  {                                     \
    cudaError_t e = (func);             \
    if (e != cudaSuccess) {             \
      return e;                         \
    }                                   \
  }
#endif

#define DISPATCH_USE_FP16_QK_REDUCTION(use_fp16_qk_reduction, USE_FP16_QK_REDUCTION, ...) \
  if (use_fp16_qk_reduction) {                                                            \
    FLASHINFER_ERROR("FP16_QK_REDUCTION disabled at compile time");                       \
  } else {                                                                                \
    constexpr bool USE_FP16_QK_REDUCTION = false;                                         \
    __VA_ARGS__                                                                           \
  }

#define DISPATCH_NUM_MMA_Q(num_mma_q, NUM_MMA_Q, ...)  \
  if (num_mma_q == 1) {                                \
    constexpr size_t NUM_MMA_Q = 1;                    \
    __VA_ARGS__                                        \
  } else if (num_mma_q == 2) {                         \
    constexpr size_t NUM_MMA_Q = 2;                    \
    __VA_ARGS__                                        \
  } else {                                             \
    std::ostringstream err_msg;                        \
    err_msg << "Unsupported num_mma_q: " << num_mma_q; \
    FLASHINFER_ERROR(err_msg.str());                   \
  }

#define DISPATCH_NUM_MMA_KV(CTA_TILE_Q, max_mma_kv, NUM_MMA_KV, ...) \
  if constexpr (CTA_TILE_Q == 128) {                                 \
    constexpr size_t NUM_MMA_KV = 4;                                 \
    __VA_ARGS__                                                      \
  } else if constexpr (CTA_TILE_Q == 64) {                           \
    if (max_mma_kv >= 4) {                                           \
      constexpr size_t NUM_MMA_KV = 4;                               \
      __VA_ARGS__                                                    \
    } else if (max_mma_kv >= 2) {                                    \
      constexpr size_t NUM_MMA_KV = 2;                               \
      __VA_ARGS__                                                    \
    } else {                                                         \
      std::ostringstream err_msg;                                    \
      err_msg << "Unsupported max_mma_kv: " << max_mma_kv;           \
      FLASHINFER_ERROR(err_msg.str());                               \
    }                                                                \
  } else if constexpr (CTA_TILE_Q == 16) {                           \
    constexpr size_t NUM_MMA_KV = 1;                                 \
    __VA_ARGS__                                                      \
  }

#define DISPATCH_CTA_TILE_Q(cta_tile_q, CTA_TILE_Q, ...)   \
  switch (cta_tile_q) {                                    \
    case 128: {                                            \
      constexpr uint32_t CTA_TILE_Q = 128;                 \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    case 64: {                                             \
      constexpr uint32_t CTA_TILE_Q = 64;                  \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    case 16: {                                             \
      constexpr uint32_t CTA_TILE_Q = 16;                  \
      __VA_ARGS__                                          \
      break;                                               \
    }                                                      \
    default: {                                             \
      std::ostringstream err_msg;                          \
      err_msg << "Unsupported cta_tile_q: " << cta_tile_q; \
      FLASHINFER_ERROR(err_msg.str());                     \
    }                                                      \
  }

#define DISPATCH_MMA_KV_AND_WARPS_Q(CTA_TILE_Q, arch, NUM_WARPS_Q, NUM_MMA_KV, ...) \
  if constexpr (CTA_TILE_Q == 128) {                                                \
    if (arch >= 1500) {                                                             \
      constexpr size_t NUM_WARPS_Q = 4;                                             \
      constexpr size_t NUM_MMA_KV = 4;                                              \
      __VA_ARGS__                                                                   \
    } else {                                                                        \
      constexpr size_t NUM_WARPS_Q = 8;                                             \
      constexpr size_t NUM_MMA_KV = 4;                                              \
      __VA_ARGS__                                                                   \
    }                                                                               \
  } else if constexpr (CTA_TILE_Q == 64) {                                          \
    constexpr size_t NUM_WARPS_Q = 4;                                               \
    constexpr size_t NUM_MMA_KV = 4;                                                \
    __VA_ARGS__                                                                     \
  } else if constexpr (CTA_TILE_Q == 16) {                                          \
    constexpr size_t NUM_WARPS_Q = 1;                                               \
    constexpr size_t NUM_MMA_KV = 1;                                                \
    __VA_ARGS__                                                                     \
  }

#define DISPATCH_GQA_GROUP_SIZE(group_size, GROUP_SIZE, ...) \
  if (group_size == 1) {                                     \
    constexpr size_t GROUP_SIZE = 1;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 2) {                              \
    constexpr size_t GROUP_SIZE = 2;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 3) {                              \
    constexpr size_t GROUP_SIZE = 3;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 4) {                              \
    constexpr size_t GROUP_SIZE = 4;                         \
    __VA_ARGS__                                              \
  } else if (group_size == 8) {                              \
    constexpr size_t GROUP_SIZE = 8;                         \
    __VA_ARGS__                                              \
  } else {                                                   \
    std::ostringstream err_msg;                              \
    err_msg << "Unsupported group_size: " << group_size;     \
    FLASHINFER_ERROR(err_msg.str());                         \
  }

#define DISPATCH_MASK_MODE(mask_mode, MASK_MODE, ...)         \
  switch (mask_mode) {                                        \
    case MaskMode::kNone: {                                   \
      constexpr MaskMode MASK_MODE = MaskMode::kNone;         \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    case MaskMode::kCausal: {                                 \
      constexpr MaskMode MASK_MODE = MaskMode::kCausal;       \
      __VA_ARGS__                                             \
      break;                                                  \
    }                                                         \
    default: {                                                \
      std::ostringstream err_msg;                             \
      err_msg << "Unsupported mask_mode: " << int(mask_mode); \
      FLASHINFER_ERROR(err_msg.str());                        \
    }                                                         \
  }

// convert head_dim to compile-time constant
#define DISPATCH_HEAD_DIM(head_dim, HEAD_DIM, ...)     \
  switch (head_dim) {                                  \
    case 64: {                                         \
      constexpr size_t HEAD_DIM = 64;                  \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 128: {                                        \
      constexpr size_t HEAD_DIM = 128;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 256: {                                        \
      constexpr size_t HEAD_DIM = 256;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    case 512: {                                        \
      constexpr size_t HEAD_DIM = 512;                 \
      __VA_ARGS__                                      \
      break;                                           \
    }                                                  \
    default: {                                         \
      std::ostringstream err_msg;                      \
      err_msg << "Unsupported head_dim: " << head_dim; \
      FLASHINFER_ERROR(err_msg.str());                 \
    }                                                  \
  }

#define DISPATCH_POS_ENCODING_MODE(pos_encoding_mode, POS_ENCODING_MODE, ...)    \
  switch (pos_encoding_mode) {                                                   \
    case PosEncodingMode::kNone: {                                               \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kNone;      \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kRoPELlama: {                                          \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kRoPELlama; \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    case PosEncodingMode::kALiBi: {                                              \
      constexpr PosEncodingMode POS_ENCODING_MODE = PosEncodingMode::kALiBi;     \
      __VA_ARGS__                                                                \
      break;                                                                     \
    }                                                                            \
    default: {                                                                   \
      std::ostringstream err_msg;                                                \
      err_msg << "Unsupported pos_encoding_mode: " << int(pos_encoding_mode);    \
      FLASHINFER_ERROR(err_msg.str());                                           \
    }                                                                            \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      std::ostringstream err_msg;                                          \
      err_msg << "Unsupported aligned_vec_size: " << aligned_vec_size;     \
      FLASHINFER_ERROR(err_msg.str());                                     \
    }                                                                      \
  }

#define DISPATCH_COMPUTE_CAP_DECODE_NUM_STAGES_SMEM(compute_capacity, NUM_STAGES_SMEM, ...) \
  if (compute_capacity.first >= 8) {                                                        \
    constexpr uint32_t NUM_STAGES_SMEM = 2;                                                 \
    __VA_ARGS__                                                                             \
  } else {                                                                                  \
    constexpr uint32_t NUM_STAGES_SMEM = 1;                                                 \
    __VA_ARGS__                                                                             \
  }

#define DISPATCH_DECODE_NUM_STAGES_SMEM(double_buff, NUM_STAGES_SMEM, ...) \
  if (double_buff) {                                                       \
    constexpr uint32_t NUM_STAGES_SMEM = 2;                                \
    __VA_ARGS__                                                            \
  } else {                                                                 \
    constexpr uint32_t NUM_STAGES_SMEM = 1;                                \
    __VA_ARGS__                                                            \
  }

namespace flashinfer {

template <typename T1, typename T2>
__forceinline__ __device__ __host__ T1 ceil_div(const T1 x, const T2 y) {
  return (x + y - 1) / y;
}

inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

template <typename T>
inline void DebugPrintCUDAArray(T* device_ptr, size_t size, std::string prefix = "") {
  std::vector<T> host_array(size);
  std::cout << prefix;
  cudaMemcpy(host_array.data(), device_ptr, size * sizeof(T), cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < size; ++i) {
    std::cout << host_array[i] << " ";
  }
  std::cout << std::endl;
}

inline uint32_t FA2DetermineCtaTileQ(int64_t avg_packed_qo_len, bool is_mla) {
  if (is_mla) {
    return 128;
  } else {
    return 64;
  }

  // if (avg_packed_qo_len > 64 && head_dim < 256) {
  //   return 128;
  // } else {
  //   auto compute_capacity = GetCudaComputeCapability();
  //   if (compute_capacity.first >= 8) {
  //     // Ampere or newer
  //     if (avg_packed_qo_len > 16) {
  //       // avg_packed_qo_len <= 64
  //       return 64;
  //     } else {
  //       // avg_packed_qo_len <= 16
  //       return 16;
  //     }
  //   } else {
  //     // NOTE(Zihao): not enough shared memory on Turing for 1x4 warp layout
  //     return 64;
  //   }
  // }
}

inline int GetSharedMemorySize() {
  int device;
  int smem_limit_per_sm;
  FLASHINFER_CUDA_CALL(cudaGetDevice(&device));
  FLASHINFER_CUDA_CALL(cudaDeviceGetAttribute(&smem_limit_per_sm,
                                              cudaDevAttrMaxSharedMemoryPerMultiprocessor, device));
  return smem_limit_per_sm;
}

inline int GetArch() {
  int deviceId{};
  FLASHINFER_CUDA_CALL(cudaGetDevice(&deviceId));
  cudaDeviceProp dprops;
  FLASHINFER_CUDA_CALL(cudaGetDeviceProperties(&dprops, deviceId));
  return dprops.major * 100 + dprops.minor;
}

/*!
 * \brief Return x - y if x > y, otherwise return 0.
 */
__device__ __forceinline__ uint32_t sub_if_greater_or_zero(uint32_t x, uint32_t y) {
  return (x > y) ? x - y : 0U;
}

__device__ __forceinline__ void swap(uint32_t& a, uint32_t& b) {
  uint32_t tmp = a;
  a = b;
  b = tmp;
}

__device__ __forceinline__ uint32_t dim2_offset(const uint32_t& dim_a, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return idx_b * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim3_offset(const uint32_t& dim_b, const uint32_t& dim_a,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return (idx_c * dim_b + idx_b) * dim_a + idx_a;
}

__device__ __forceinline__ uint32_t dim4_offset(const uint32_t& dim_c, const uint32_t& dim_b,
                                                const uint32_t& dim_a, const uint32_t& idx_d,
                                                const uint32_t& idx_c, const uint32_t& idx_b,
                                                const uint32_t& idx_a) {
  return ((idx_d * dim_c + idx_c) * dim_b + idx_b) * dim_a + idx_a;
}

#define DEFINE_HAS_MEMBER(member)                                                              \
  template <typename T, typename = void>                                                       \
  struct has_##member : std::false_type {};                                                    \
  template <typename T>                                                                        \
  struct has_##member<T, std::void_t<decltype(std::declval<T>().member)>> : std::true_type {}; \
  template <typename T>                                                                        \
  inline constexpr bool has_##member##_v = has_##member<T>::value;

__forceinline__ __device__ void sync_threads() {
  __builtin_mxc_arrive_bsmcnt(0);
  __builtin_mxc_barrier_ex(4);
}

template <int N = 0, int M = 4>
__forceinline__ __device__ void sync_threads() {
  __builtin_mxc_arrive_bsmcnt(N);
  __builtin_mxc_barrier_ex(M);
}

// used for ldg_bsm
template <int N>
__forceinline__ __device__ void cp_async_bsm_wait() {
  __builtin_mxc_arrive_gvmcnt(N);
  __builtin_mxc_barrier_ex(4);
}

__forceinline__ __device__ void permute_64bx4(uint32_t (*src)[2], uint32_t (*dst)[2]) {
  dst[0][0] = __builtin_mxc_byte_perm(src[1][0], src[0][0], 0x05040100);
  dst[1][0] = __builtin_mxc_byte_perm(src[1][0], src[0][0], 0x07060302);
  dst[2][0] = __builtin_mxc_byte_perm(src[1][1], src[0][1], 0x05040100);
  dst[3][0] = __builtin_mxc_byte_perm(src[1][1], src[0][1], 0x07060302);
  dst[0][1] = __builtin_mxc_byte_perm(src[3][0], src[2][0], 0x05040100);
  dst[1][1] = __builtin_mxc_byte_perm(src[3][0], src[2][0], 0x07060302);
  dst[2][1] = __builtin_mxc_byte_perm(src[3][1], src[2][1], 0x05040100);
  dst[3][1] = __builtin_mxc_byte_perm(src[3][1], src[2][1], 0x07060302);
}

__forceinline__ __device__ void permute_64bx4(uint32_t(*src), uint32_t (*dst)[2]) {
  dst[0][0] = __builtin_mxc_byte_perm(src[2], src[0], 0x05040100);
  dst[1][0] = __builtin_mxc_byte_perm(src[2], src[0], 0x07060302);
  dst[2][0] = __builtin_mxc_byte_perm(src[3], src[1], 0x05040100);
  dst[3][0] = __builtin_mxc_byte_perm(src[3], src[1], 0x07060302);
  dst[0][1] = __builtin_mxc_byte_perm(src[6], src[4], 0x05040100);
  dst[1][1] = __builtin_mxc_byte_perm(src[6], src[4], 0x07060302);
  dst[2][1] = __builtin_mxc_byte_perm(src[7], src[5], 0x05040100);
  dst[3][1] = __builtin_mxc_byte_perm(src[7], src[5], 0x07060302);
}

__forceinline__ __device__ void permute_128bx4(uint32_t (*src)[4], uint32_t (*dst)[2],
                                               uint32_t GROUP_ID) {
  dst[0][0] =
      __builtin_mxc_byte_perm(src[1][0 + GROUP_ID * 2], src[0][0 + GROUP_ID * 2], 0x05040100);
  dst[1][0] =
      __builtin_mxc_byte_perm(src[1][0 + GROUP_ID * 2], src[0][0 + GROUP_ID * 2], 0x07060302);
  dst[2][0] =
      __builtin_mxc_byte_perm(src[1][1 + GROUP_ID * 2], src[0][1 + GROUP_ID * 2], 0x05040100);
  dst[3][0] =
      __builtin_mxc_byte_perm(src[1][1 + GROUP_ID * 2], src[0][1 + GROUP_ID * 2], 0x07060302);
  dst[0][1] =
      __builtin_mxc_byte_perm(src[3][0 + GROUP_ID * 2], src[2][0 + GROUP_ID * 2], 0x05040100);
  dst[1][1] =
      __builtin_mxc_byte_perm(src[3][0 + GROUP_ID * 2], src[2][0 + GROUP_ID * 2], 0x07060302);
  dst[2][1] =
      __builtin_mxc_byte_perm(src[3][1 + GROUP_ID * 2], src[2][1 + GROUP_ID * 2], 0x05040100);
  dst[3][1] =
      __builtin_mxc_byte_perm(src[3][1 + GROUP_ID * 2], src[2][1 + GROUP_ID * 2], 0x07060302);
}

template <typename T, int SIZE>
__forceinline__ __device__ void clear(T* frag) {
#pragma unroll
  for (uint32_t i = 0; i < SIZE; ++i) {
    frag[i] = 0;
  }
}

// output[0] = a[0] * b[0] + c[0], output[1] = a[1] * b[1] + c[1]
__forceinline__ __device__ void fma_f32x2(float* output, const float* a, const float* b, float* c) {
  typedef __NATIVE_VECTOR__(2, float) Float2;
  Float2 vec_a = {a[0], a[1]};
  Float2 vec_b = {b[0], b[1]};
  Float2 vec_c = {c[0], c[1]};
  Float2 vec_o = __builtin_mxc_pk_fma_f32(vec_a, vec_b, vec_c);
  *(Float2*)output = vec_o;
}

// output[0] = a[0] * b[0], output[1] = a[1] * b[1]
__forceinline__ __device__ void fma_f32x2(float* output, const float* a, const float* b) {
  typedef __NATIVE_VECTOR__(2, float) Float2;
  Float2 vec_a = {a[0], a[1]};
  Float2 vec_b = {b[0], b[1]};
  Float2 vec_c = {0.f, 0.f};
  Float2 vec_o = __builtin_mxc_pk_fma_f32(vec_a, vec_b, vec_c);
  *(Float2*)output = vec_o;
}

// output[0] = a[0] * scale, output[1] = a[1] * scale
__forceinline__ __device__ void fma_f32x2(float* output, const float* a, const float scale,
                                          const float c = 0) {
  typedef __NATIVE_VECTOR__(2, float) Float2;
  Float2 vec_a = {a[0], a[1]};
  Float2 vec_b = {scale, scale};
  Float2 vec_c = {c, c};
  Float2 vec_o = __builtin_mxc_pk_fma_f32(vec_a, vec_b, vec_c);
  *(Float2*)output = vec_o;
}

}  // namespace flashinfer

#endif  // FLASHINFER_UTILS_CUH_
