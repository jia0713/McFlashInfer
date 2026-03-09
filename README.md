McFlashInfer is a library and kernel generator for Large Language Models that provides high-performance implementation of LLM GPU kernels such as FlashAttention, SparseAttention, PageAttention, Sampling, and more on MACA platform. McFlashInfer focuses on LLM serving and inference, and delivers state-of-the-art performance across diverse scenarios.

## Create Conda Env
```
conda create -n your_env_name python=3.10
```

## Activate Conda Env
```
conda activate your_env_name
```

## Install Dependencies
```
pip install your_maca_torch2.4/2.6_whl --force-reinstall --no-deps
pip install your_maca_triton.whl --force-reinstall --no-deps
pip install ninja
pip install einops
pip install setuptools==75.8.2
pip install numpy==1.24.2
pip install pytest
pip install packaging
pip install SentencePiece
pip install accelerate
pip install wheel
pip install build
pip install black
pip install cpplint
pip install pylint
```

## Set Environment Variables
```
export MACA_PATH=/your/maca/path
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/mxgpu_llvm/lib:$LD_LIBRARY_PATH
export CUDA_PATH=$MACA_PATH/tools/cu-bridge
export PATH=$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
```

## Build
Clean build artifacts if needed.
```
./clean.sh
```

Build AOT kernels and create FlashInfer distributions.
``` sh
python -m flashinfer.aot
python -m build --no-isolation --wheel
```

Please don't use JIT mode because it is not stable yet.

## Install Wheel
```
pip install dist/flashinfer-*.whl
```
