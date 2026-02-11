#!/bin/bash

# OriginDL Build Script
# Usage: ./build.sh [ORIGIN|origin|TORCH|torch] [--cuda] [--nvcc /path/to/nvcc] [--libtorch_path /path/to/libtorch] [--build_dir /path/to/build]

set -e

BACKEND=${1:-ORIGIN}
ENABLE_CUDA=false
NVCC_PATH="nvcc"
LIBTORCH_PATH=""
BUILD_DIR=""
# NVCC_PATH="/usr/local/cuda/bin/nvcc"  # 默认CUDA路径

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
        --libtorch_path)
            LIBTORCH_PATH="$2"
            shift 2
            ;;
        --build_dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [ORIGIN|origin|TORCH|torch] [--cuda] [--nvcc /path/to/nvcc] [--libtorch_path /path/to/libtorch] [--build_dir /path/to/build]"
            exit 1
            ;;
    esac
done

# Convert backend to uppercase for consistency
BACKEND=$(echo "$BACKEND" | tr '[:lower:]' '[:upper:]')

# Check backend parameter
if [ "$BACKEND" != "TORCH" ] && [ "$BACKEND" != "ORIGIN" ]; then
    echo "Error: Invalid backend parameter '$1'"
    echo "Usage: $0 [ORIGIN|origin|TORCH|torch] [--cuda] [--nvcc /path/to/nvcc] [--libtorch_path /path/to/libtorch] [--build_dir /path/to/build]"
    exit 1
fi

# Auto-detect CUDA support if not explicitly enabled
if [ "$ENABLE_CUDA" != true ]; then
    # Check if nvcc is available and get its path
    NVCC_FOUND=$(which nvcc 2>/dev/null)
    if [ -n "$NVCC_FOUND" ]; then
        # Check if CUDA libraries are available
        # Try common CUDA library paths
        if [ -f "/usr/local/cuda/lib64/libcudart.so" ] || \
           [ -f "/usr/lib/x86_64-linux-gnu/libcudart.so" ] || \
           ldconfig -p 2>/dev/null | grep -q libcudart; then
            ENABLE_CUDA=true
            NVCC_PATH="$NVCC_FOUND"
            echo "CUDA detected automatically, enabling CUDA support (nvcc: $NVCC_PATH)"
        fi
    fi
fi

echo "OriginDL using backend: $BACKEND"
# Set LibTorch path
if [ "$BACKEND" = "TORCH" ]; then
    # 优先级：命令行参数 > 环境变量 > 默认路径
    if [ -n "$LIBTORCH_PATH" ]; then
        export TORCH_PATH="$LIBTORCH_PATH"
    elif [ -z "$TORCH_PATH" ]; then
        export TORCH_PATH="$(pwd)/3rd/libtorch"
    fi
    echo "LibTorch Path: $TORCH_PATH"
    
    # Check if LibTorch exists
    if [ ! -d "$TORCH_PATH/include" ] || [ ! -d "$TORCH_PATH/lib" ]; then
        echo "Error: LibTorch not found at $TORCH_PATH"
        echo "Please specify the correct path using --libtorch_path option or TORCH_PATH environment variable"
        exit 1
    fi
elif [ "$BACKEND" = "ORIGIN" ]; then
    echo "Using OriginMat backend (no external dependencies)"
else
    echo "Error: Other backends are not supported"
    exit 1
fi

# 根据后端选择构建目录（如果未指定）
if [ -z "$BUILD_DIR" ]; then
    if [ "$BACKEND" = "ORIGIN" ]; then
        BUILD_DIR="build"
    elif [ "$BACKEND" = "TORCH" ]; then
        BUILD_DIR="torch_build"
    else
        BUILD_DIR="other_build"
    fi
fi

echo "Building $BACKEND backend in: $BUILD_DIR"

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
    COMPILE_COMMANDS_PATH="$BUILD_DIR/compile_commands.json"
    if [ -f "$COMPILE_COMMANDS_PATH" ]; then
        echo "Creating symbolic link to compile_commands.json..."
        ln -s "$COMPILE_COMMANDS_PATH" compile_commands.json
        echo "Symbolic link created successfully."
    fi
else
    #echo "compile_commands.json already exists, skipping symbolic link creation."
    :
fi
