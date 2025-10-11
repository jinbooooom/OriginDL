#!/bin/bash

# OriginDL Build Script
# Usage: ./build.sh [ARRAYFIRE|TORCH]

set -e

BACKEND=${1:-TORCH}

# Check backend parameter
if [ "$BACKEND" != "ARRAYFIRE" ] && [ "$BACKEND" != "TORCH" ]; then
    echo "Error: Invalid backend parameter '$BACKEND'"
    echo "Usage: $0 [ARRAYFIRE|TORCH]"
    exit 1
fi

echo "OriginDL using backend: $BACKEND"
# Set LibTorch path
if [ "$BACKEND" = "TORCH" ]; then
    if [ -z "$TORCH_PATH" ]; then
        export TORCH_PATH="$(pwd)/3rd/libtorch"
    fi
    echo "LibTorch Path: $TORCH_PATH"
    
    # Check if LibTorch exists
    if [ ! -d "$TORCH_PATH/include" ] || [ ! -d "$TORCH_PATH/lib" ]; then
        echo "Error: LibTorch not found at $TORCH_PATH"
        exit 1
    fi
else
    echo "Error: Other backends are not supported"
    exit 1
fi

mkdir -p build
cd build

# 配置项目并启用编译命令导出 compile_commands.json, 保证 vscode 可以跳转
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DMAT_BACKEND=$BACKEND ..
time make -j`nproc`
cd ../

# 在项目根目录创建符号链接。检查目标文件是否存在，如果不存在才创建符号链接
if [ ! -e compile_commands.json ]; then
    echo "Creating symbolic link to compile_commands.json..."
    ln -s build/compile_commands.json .
    echo "Symbolic link created successfully."
else
    #echo "compile_commands.json already exists, skipping symbolic link creation."
    :
fi
