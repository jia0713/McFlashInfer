/*
 * 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
 *
 * Copyright (c) 2024 by FlashInfer team.
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
#ifndef FLASHINFER_CUTLASS_UTILS_CUH_
#define FLASHINFER_CUTLASS_UTILS_CUH_

#include "cute/tensor.hpp"
#include "mctlass/epilogue/collective/collective_builder.hpp"
#include "mctlass/epilogue/collective/default_epilogue.hpp"
#include "mctlass/epilogue/thread/linear_combination.h"
#include "mctlass/gemm/collective/collective_builder.hpp"
#include "mctlass/gemm/device/gemm_grouped.h"
#include "mctlass/gemm/device/gemm_universal_adapter.h"
#include "mctlass/gemm/dispatch_policy.hpp"
// #include "mctlass/gemm/group_array_problem_shape.hpp"
#include "mctlass/gemm/kernel/default_gemm_grouped.h"
#include "mctlass/gemm/kernel/gemm_universal.hpp"
#include "mctlass/layout/matrix.h"
#include "mctlass/mctlass.h"
#include "mctlass/numeric_types.h"
#include "mctlass/tensor_ref.h"
#include "mctlass/util/command_line.h"
#include "mctlass/util/distribution.h"
#include "mctlass/util/host_tensor.h"
#include "mctlass/util/packed_stride.hpp"
#include "mctlass/util/reference/device/gemm.h"
#include "mctlass/util/reference/device/tensor_compare.h"
#include "mctlass/util/reference/device/tensor_fill.h"
#include "mctlass/util/tensor_view_io.h"

namespace flashinfer {

template <typename T>
struct cutlass_dtype {
  using type = T;
};

template <>
struct cutlass_dtype<half> {
  using type = mctlass::half_t;
};

template <>
struct cutlass_dtype<maca_bfloat16> {
  using type = mctlass::bfloat16_t;
};

// template <>
// struct cutlass_dtype<__nv_fp8_e4m3> {
//   using type = mctlass::float_e4m3_t;
// };

// template <>
// struct cutlass_dtype<__nv_fp8_e5m2> {
//   using type = mctlass::float_e5m2_t;
// };

template <typename T>
using cutlass_dtype_t = typename cutlass_dtype<T>::type;

template <typename T>
void compileTimeDebug(T&&) {
  static_assert(sizeof(T) == 0, "Compile time debug");
}

#define CUTLASS_CHECK(cmd)                                                            \
  do {                                                                                \
    auto status = cmd;                                                                \
    if (status != cutlass::Status::kSuccess) {                                        \
      std::ostringstream err_msg;                                                     \
      err_msg << "cutlass " << #cmd << " failed: " << cutlassGetStatusString(status); \
      FLASHINFER_ERROR(err_msg.str());                                                \
    }                                                                                 \
  } while (0)

}  // namespace flashinfer

#endif  // FLASHINFER_CUTLASS_UTILS_CUH_
