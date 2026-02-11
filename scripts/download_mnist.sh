#!/bin/bash

# MNIST 数据集下载脚本
# 用法: download_mnist.sh [-d DIR] [--dir DIR]
# 默认下载到 ./data/mnist/

set -e

DATA_DIR="./data/mnist"

# 解析选项：-d DIR / --dir DIR 指定下载目录
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -d/--dir requires a directory argument." >&2
                exit 1
            fi
            DATA_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-d DIR] [--dir DIR]"
            echo "  -d, --dir DIR   Download MNIST to DIR (default: ./data/mnist)"
            echo "  -h, --help      Show this help"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1. Use -h or --help for usage." >&2
            exit 1
            ;;
    esac
done

# 使用 Google Cloud Storage 镜像（更可靠）
BASE_URL="https://storage.googleapis.com/cvdf-datasets/mnist"

# 创建数据目录
mkdir -p "$DATA_DIR"

echo "Downloading MNIST dataset to $DATA_DIR..."

# 检查下载工具
if command -v wget > /dev/null 2>&1; then
    DOWNLOAD_CMD="wget"
    WGET_OPTS="-q --show-progress"
elif command -v curl > /dev/null 2>&1; then
    DOWNLOAD_CMD="curl"
    CURL_OPTS="-L --progress-bar"
else
    echo "Error: Neither wget nor curl is installed. Please install one of them." >&2
    exit 1
fi

# 下载训练集
echo "Downloading training images..."
if [ "$DOWNLOAD_CMD" = "wget" ]; then
    wget $WGET_OPTS "$BASE_URL/train-images-idx3-ubyte.gz" -O "$DATA_DIR/train-images-idx3-ubyte.gz" || exit 1
else
    curl $CURL_OPTS "$BASE_URL/train-images-idx3-ubyte.gz" -o "$DATA_DIR/train-images-idx3-ubyte.gz" || exit 1
fi

echo "Downloading training labels..."
if [ "$DOWNLOAD_CMD" = "wget" ]; then
    wget $WGET_OPTS "$BASE_URL/train-labels-idx1-ubyte.gz" -O "$DATA_DIR/train-labels-idx1-ubyte.gz" || exit 1
else
    curl $CURL_OPTS "$BASE_URL/train-labels-idx1-ubyte.gz" -o "$DATA_DIR/train-labels-idx1-ubyte.gz" || exit 1
fi

# 下载测试集
echo "Downloading test images..."
if [ "$DOWNLOAD_CMD" = "wget" ]; then
    wget $WGET_OPTS "$BASE_URL/t10k-images-idx3-ubyte.gz" -O "$DATA_DIR/t10k-images-idx3-ubyte.gz" || exit 1
else
    curl $CURL_OPTS "$BASE_URL/t10k-images-idx3-ubyte.gz" -o "$DATA_DIR/t10k-images-idx3-ubyte.gz" || exit 1
fi

echo "Downloading test labels..."
if [ "$DOWNLOAD_CMD" = "wget" ]; then
    wget $WGET_OPTS "$BASE_URL/t10k-labels-idx1-ubyte.gz" -O "$DATA_DIR/t10k-labels-idx1-ubyte.gz" || exit 1
else
    curl $CURL_OPTS "$BASE_URL/t10k-labels-idx1-ubyte.gz" -o "$DATA_DIR/t10k-labels-idx1-ubyte.gz" || exit 1
fi

# 解压文件
echo "Extracting files..."
gunzip -f "$DATA_DIR/train-images-idx3-ubyte.gz" || true
gunzip -f "$DATA_DIR/train-labels-idx1-ubyte.gz" || true
gunzip -f "$DATA_DIR/t10k-images-idx3-ubyte.gz" || true
gunzip -f "$DATA_DIR/t10k-labels-idx1-ubyte.gz" || true

echo "MNIST dataset downloaded and extracted successfully!"
echo "Files are in: $DATA_DIR"

