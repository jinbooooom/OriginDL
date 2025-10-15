#!/bin/bash

# OriginDL Build Script
# Usage: ./build.sh [ORIGIN|origin|TORCH|torch]

set -e

BACKEND=${1:-ORIGIN}

# Convert backend to uppercase for consistency
BACKEND=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')

# Check backend parameter
if [ "$BACKEND" != "TORCH" ] && [ "$BACKEND" != "ORIGIN" ]; then
    echo "Error: Invalid backend parameter '$1'"
    echo "Usage: $0 [ORIGIN|origin|TORCH|torch]"
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
elif [ "$BACKEND" = "ORIGIN" ]; then
    echo "Using OriginMat backend (no external dependencies)"
else
    echo "Error: Other backends are not supported"
    exit 1
fi

# 根据后端选择不同的构建目录
if [ "$BACKEND" = "ORIGIN" ]; then
    BUILD_DIR="build"
elif [ "$BACKEND" = "TORCH" ]; then
    BUILD_DIR="torch_build"
else
    BUILD_DIR="other_build"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

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
