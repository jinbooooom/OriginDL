#!/bin/bash

# OriginDL Build Script
# Usage: ./build.sh [ORIGIN|origin|TORCH|torch] [--cuda] [--nvcc /path/to/nvcc]

set -e

BACKEND=${1:-ORIGIN}
ENABLE_CUDA=false
NVCC_PATH="/usr/local/cuda-12.8/bin/nvcc"  # 默认CUDA路径

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        ORIGIN|origin|TORCH|torch)
            BACKEND=$1
            shift
            ;;
        --cuda)
            ENABLE_CUDA=true
            shift
            ;;
        --nvcc)
            NVCC_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [ORIGIN|origin|TORCH|torch] [--cuda] [--nvcc /path/to/nvcc]"
            exit 1
            ;;
    esac
done

# Convert backend to uppercase for consistency
BACKEND=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')

# Check backend parameter
if [ "$BACKEND" != "TORCH" ] && [ "$BACKEND" != "ORIGIN" ]; then
    echo "Error: Invalid backend parameter '$1'"
    echo "Usage: $0 [ORIGIN|origin|TORCH|torch] [--cuda]"
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

# 根据后端选择构建目录
if [ "$BACKEND" = "ORIGIN" ]; then
    BUILD_DIR="build"
    echo "Building OriginMat backend in: $BUILD_DIR"
elif [ "$BACKEND" = "TORCH" ]; then
    BUILD_DIR="torch_build"
    echo "Building LibTorch backend in: $BUILD_DIR"
else
    BUILD_DIR="other_build"
    echo "Building in default directory: $BUILD_DIR"
fi

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# 设置CMake参数
CMAKE_ARGS="-DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DMAT_BACKEND=$BACKEND"

if [ "$ENABLE_CUDA" = true ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CUDA=ON"
    # 设置CUDA编译器路径
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_CUDA_COMPILER=$NVCC_PATH"
    echo "CUDA support enabled (using nvcc: $NVCC_PATH)"
fi

# 配置和编译
echo "Configuring with: $CMAKE_ARGS"
cmake $CMAKE_ARGS ..
time make -j$(nproc)

echo "Build completed! All artifacts are in $BUILD_DIR directory"

cd ..

# 在项目根目录创建符号链接。检查目标文件是否存在，如果不存在才创建符号链接
if [ ! -e compile_commands.json ]; then
    echo "Creating symbolic link to compile_commands.json..."
    ln -s build/compile_commands.json .
    echo "Symbolic link created successfully."
else
    #echo "compile_commands.json already exists, skipping symbolic link creation."
    :
fi
