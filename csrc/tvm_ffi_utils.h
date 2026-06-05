/*
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
#pragma once

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/dtype.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

#include <dlpack/dlpack.h>
#include <cuda_runtime.h>

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;
namespace ffi = tvm::ffi;

namespace tvm {
namespace ffi {

class CUDADeviceGuard {
 public:
  explicit CUDADeviceGuard(int device_id) {
    cudaGetDevice(&previous_device_id_);
    if (previous_device_id_ != device_id) {
      cudaSetDevice(device_id);
      changed_ = true;
    }
  }

  ~CUDADeviceGuard() {
    if (changed_) {
      cudaSetDevice(previous_device_id_);
    }
  }

 private:
  int previous_device_id_{0};
  bool changed_{false};
};

}  // namespace ffi
}  // namespace tvm

inline constexpr int64_t encode_dlpack_dtype(DLDataType dtype) {
  return (dtype.code << 16) | (dtype.bits << 8) | dtype.lanes;
}

inline size_t get_element_size(TensorView tensor) {
  DLDataType dtype = tensor.dtype();
  return (dtype.bits * dtype.lanes + 7) / 8;
}

inline cudaStream_t get_stream(DLDevice device) {
  return static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));
}

constexpr DLDataType dl_uint8 = DLDataType{kDLUInt, 8, 1};
constexpr DLDataType dl_uint16 = DLDataType{kDLUInt, 16, 1};
constexpr DLDataType dl_uint32 = DLDataType{kDLUInt, 32, 1};
constexpr DLDataType dl_uint64 = DLDataType{kDLUInt, 64, 1};
constexpr DLDataType dl_int8 = DLDataType{kDLInt, 8, 1};
constexpr DLDataType dl_int16 = DLDataType{kDLInt, 16, 1};
constexpr DLDataType dl_int32 = DLDataType{kDLInt, 32, 1};
constexpr DLDataType dl_int64 = DLDataType{kDLInt, 64, 1};
constexpr DLDataType dl_float16 = DLDataType{kDLFloat, 16, 1};
constexpr DLDataType dl_float32 = DLDataType{kDLFloat, 32, 1};
constexpr DLDataType dl_float64 = DLDataType{kDLFloat, 64, 1};
constexpr DLDataType dl_bfloat16 = DLDataType{kDLBfloat, 16, 1};
constexpr DLDataType dl_bool = DLDataType{kDLBool, 8, 1};

constexpr int64_t float16_code = encode_dlpack_dtype(dl_float16);
constexpr int64_t bfloat16_code = encode_dlpack_dtype(dl_bfloat16);
constexpr int64_t float32_code = encode_dlpack_dtype(dl_float32);
constexpr int64_t uint8_code = encode_dlpack_dtype(dl_uint8);
constexpr int64_t int32_code = encode_dlpack_dtype(dl_int32);
constexpr int64_t int64_code = encode_dlpack_dtype(dl_int64);

#define DISPATCH_BOOL(expr, const_expr, ...) \
  [&]() -> bool {                            \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      return __VA_ARGS__();                  \
    } else {                                 \
      constexpr bool const_expr = false;     \
      return __VA_ARGS__();                  \
    }                                        \
  }()
