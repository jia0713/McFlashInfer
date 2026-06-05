"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# NOTE(lequn): Do not "from .jit.env import xxx".
# Do "from .jit import env as jit_env" and use "jit_env.xxx" instead.
# This helps AOT script to override envs.

import os
import pathlib
import re
import warnings

from torch.utils.cpp_extension import _get_cuda_arch_flags


def _get_workspace_dir_name() -> pathlib.Path:
    flashinfer_base = os.getenv(
        "FLASHINFER_WORKSPACE_BASE", pathlib.Path.home().as_posix()
    )
    return pathlib.Path(flashinfer_base) / ".cache" / "flashinfer" / "80"


# use pathlib
FLASHINFER_WORKSPACE_DIR = _get_workspace_dir_name()
FLASHINFER_JIT_DIR = FLASHINFER_WORKSPACE_DIR / "cached_ops"
FLASHINFER_GEN_SRC_DIR = FLASHINFER_WORKSPACE_DIR / "generated"
_package_root = pathlib.Path(__file__).resolve().parents[1]
_data_root = _package_root / "data"
_source_tree_mode = not (_data_root / "tvm_binding").exists()
if _source_tree_mode:
    _data_root = _package_root.parent
FLASHINFER_DATA = _data_root
FLASHINFER_INCLUDE_DIR = _data_root / "include"
FLASHINFER_CSRC_DIR = _data_root / "csrc"
# FLASHINFER_SRC_DIR = _data_root / "src"
FLASHINFER_TVM_BINDING_DIR = _data_root / "tvm_binding"
FLASHINFER_AOT_DIR = _data_root / "aot"
CUTLASS_INCLUDE_DIRS = [
    (
        _data_root / "3rdparty" / "mcTlass" / "include"
        if _source_tree_mode
        else _data_root / "cutlass" / "include"
    ),
    (
        _data_root / "3rdparty" / "mcTlass" / "tools" / "util" / "include"
        if _source_tree_mode
        else _data_root / "cutlass" / "tools" / "util" / "include"
    ),
]
