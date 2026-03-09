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
#ifndef FLASHINFER_MATH_CUH_
#define FLASHINFER_MATH_CUH_

#include <maca_fp16.h>
#include <mc_runtime.h>

#include <cstdint>

namespace flashinfer {
namespace math {

// log2(e)
constexpr float log2e = 1.44269504088896340736f;

constexpr float loge2 = 0.693147180559945309417f;

constexpr float inf = 5e4;

__forceinline__ __device__ half2 uint32_as_half2(uint32_t x) { return *(half2*)&x; }

__forceinline__ __device__ uint32_t half2_as_uint32(half2 x) { return *(uint32_t*)&x; }

/*!
 * \brief Wrapper of PTX ex2.approx instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ float ptx_exp2(float x) {
#if defined(CHECK_NANS)
  float (*__ftz)(const float) = [](const float in) {
    float res = in;
    if (((unsigned int&)in & 0x7f800000) == 0 && (int&)in & 0x007fffff) {
      (unsigned int&)res = (unsigned int&)in & 0x80000000;
    }
    return res;
  };
  x = __ftz(x);
  float y = exp2f(x);
  y = __ftz(y);
  return y;
#else
  float y = __builtin_exp2f(x);
  return y;
#endif
}

/*!
 * \brief Wrapper of PTX lg2.approx instruction, which computes log2(x)
 * \param x input
 */
__forceinline__ __device__ float ptx_log2(float x) {
#if defined(CHECK_NANS)
  float (*__ftz)(const float) = [](const float in) {
    float res = in;
    if (((unsigned int&)in & 0x7f800000) == 0 && (int&)in & 0x007fffff) {
      (unsigned int&)res = (unsigned int&)in & 0x80000000;
    }
    return res;
  };
  x = __ftz(x);
  float y = __log2f(x);
  y = __ftz(y);
  return y;
#else
  float y = __log2f(x);
  return y;
#endif
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16x2 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
  uint32_t y_u32;
  uint32_t x_u32 = half2_as_uint32(x);
  unsigned int __a = (x_u32);
  __half2 __d = h2exp2(*(__half2*)&__a);
  y_u32 = *(unsigned int*)&__d;
  return uint32_as_half2(y_u32);
}

/*!
 * \brief Wrapper of PTX ex2.approx.f16 instruction, which computes 2^x
 * \param x input
 */
__forceinline__ __device__ half ptx_exp2(half x) {
  ushort y_u16;
  unsigned short __a = (__half_as_ushort(x));
  __half __d = hexp2(*(__half*)&__a);
  y_u16 = *(unsigned short*)&__d;
  return __ushort_as_half(y_u16);
}

/*!
 * \brief Wrapper of PTX rcp.approx instruction, which computes 1/x
 * \param x input
 */
__forceinline__ __device__ float ptx_rcp(float x) {
  float y;
#if defined(CHECK_NANS)
  float (*__ftz)(const float) = [](const float in) {
    float res = in;
    if (((unsigned int&)in & 0x7f800000) == 0 && (int&)in & 0x007fffff) {
      (unsigned int&)res = (unsigned int&)in & 0x80000000;
    }
    return res;
  };
  float __a = __ftz(x);
  y = 1.f / __a;
  y = __ftz(y);
#else
  y = 1.f / x;
#endif
  return y;
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction, which performs a butterfly shuffle
 *   between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ delta]
 */
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
  return __shfl_xor_sync(uint64_t(-1), x, lane_mask);
}

/*!
 * \brief Wrapper of PTX shfl.sync.bfly instruction on half2, which performs a butterfly
 *   shuffle between threads in a warp.
 * \param x The value in the source lane
 * \param lane_mask The mask to perform thread index xor with: y[i] <- x[i ^ lane_mask]
 */
__forceinline__ __device__ half2 shfl_xor_sync(half2 x, int lane_mask) {
  return __shfl_xor_sync(uint64_t(-1), x, lane_mask);
}

/*!
 * \brief Wrapper of PTX rsqrt approximation instruction, which computes 1/sqrt(x)
 * \param x input
 */
__forceinline__ __device__ float rsqrt(float x) {
  float y;
#if defined(CHECK_NANS)
  float (*__ftz)(const float) = [](const float in) {
    float res = in;
    if (((unsigned int&)in & 0x7f800000) == 0 && (int&)in & 0x007fffff) {
      (unsigned int&)res = (unsigned int&)in & 0x80000000;
    }
    return res;
  };
  float __a = __ftz(x);
  y = rsqrtf(__a);
  y = __ftz(y);
#else
  y = rsqrtf(x);
#endif
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f32 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ float tanh(float x) {
  float y = tanhf(x);
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16x2 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half2 tanh(half2 x) {
  half2 y;
  y.x = half(tanh(float(x.x)));
  y.y = half(tanh(float(x.y)));
  return y;
}

/*!
 * \brief Wrapper of PTX tanh.approx.f16 instruction, which computes tanh(x)
 * \param x input
 */
__forceinline__ __device__ half tanh(half x) {
  half y = half(tanh(float(x)));
  return y;
}

}  // namespace math
}  // namespace flashinfer
#endif  // FLASHINFER_MATH_CUH_
