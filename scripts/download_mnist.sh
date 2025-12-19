#!/bin/bash

# MNIST 数据集下载脚本
# 下载 MNIST 数据集到 ./data/ 目录

set -e

DATA_DIR="./data"
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

