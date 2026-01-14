#!/bin/bash
export EXT_PARALLEL=1
export TORCH_CUDA_ARCH_LIST="12.0" # For Blackwell

echo "Uninstalling sageattention"
uv pip uninstall sageattention
cd ..
rm -rvf SageAttention
echo "Clone Sage Attention"
git clone https://github.com/thu-ml/SageAttention sageattention
cd sageattention
echo "Installing Sage Attention"
python3 setup.py install