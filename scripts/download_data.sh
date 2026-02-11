#!/bin/bash

# OriginDL 数据和模型下载脚本
# 从 GitHub Releases 下载数据和模型文件

set -e

# 配置
REPO_OWNER="jinbooooom"  # GitHub 用户名
REPO_NAME="OriginDL"        # 仓库名
VERSION="v1.0.0"            # Release 版本号
BASE_URL="https://github.com/${REPO_OWNER}/${REPO_NAME}/releases/download/${VERSION}"

DATA_ARCHIVE="origindl-data-${VERSION}.tar.gz"
MODEL_ARCHIVE="origindl-model-${VERSION}.tar.gz"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查下载工具
if command -v wget > /dev/null 2>&1; then
    DOWNLOAD_CMD="wget"
    WGET_OPTS="-q --show-progress"
elif command -v curl > /dev/null 2>&1; then
    DOWNLOAD_CMD="curl"
    CURL_OPTS="-L --progress-bar -f"
else
    echo -e "${RED}Error: Neither wget nor curl is installed. Please install one of them.${NC}" >&2
    exit 1
fi

# 下载函数
download_file() {
    local url=$1
    local output=$2
    local name=$3
    
    echo -e "${YELLOW}Downloading ${name}...${NC}"
    
    if [ "$DOWNLOAD_CMD" = "wget" ]; then
        if wget $WGET_OPTS "$url" -O "$output"; then
            echo -e "${GREEN}✓ ${name} downloaded successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to download ${name}${NC}" >&2
            return 1
        fi
    else
        if curl $CURL_OPTS "$url" -o "$output"; then
            echo -e "${GREEN}✓ ${name} downloaded successfully${NC}"
            return 0
        else
            echo -e "${RED}✗ Failed to download ${name}${NC}" >&2
            return 1
        fi
    fi
}

# 解压函数
extract_file() {
    local archive=$1
    local name=$2
    
    if [ ! -f "$archive" ]; then
        echo -e "${RED}Error: ${archive} not found${NC}" >&2
        return 1
    fi
    
    echo -e "${YELLOW}Extracting ${name}...${NC}"
    if tar -xzf "$archive"; then
        echo -e "${GREEN}✓ ${name} extracted successfully${NC}"
        rm -f "$archive"  # 删除压缩包以节省空间
        return 0
    else
        echo -e "${RED}✗ Failed to extract ${name}${NC}" >&2
        return 1
    fi
}

# 主函数
main() {
    echo "=========================================="
    echo "OriginDL Data and Model Download Script"
    echo "=========================================="
    echo ""
    
    # 检查是否在项目根目录
    if [ ! -f "CMakeLists.txt" ] && [ ! -f "README.md" ]; then
        echo -e "${RED}Error: Please run this script from the project root directory${NC}" >&2
        exit 1
    fi
    
    # 创建临时目录
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    cd "$TEMP_DIR"
    
    # 下载数据文件
    if download_file "${BASE_URL}/${DATA_ARCHIVE}" "$DATA_ARCHIVE" "Data archive"; then
        extract_file "$DATA_ARCHIVE" "Data files"
        # 移动数据文件到项目目录
        if [ -d "data" ]; then
            cp -r data/* "$OLDPWD/data/" 2>/dev/null || true
            echo -e "${GREEN}Data files copied to ./data/${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: Failed to download data archive. You may need to download it manually.${NC}"
    fi
    
    # 下载模型文件
    if download_file "${BASE_URL}/${MODEL_ARCHIVE}" "$MODEL_ARCHIVE" "Model archive"; then
        extract_file "$MODEL_ARCHIVE" "Model files"
        # 移动模型文件到项目目录
        if [ -d "model" ]; then
            cp -r model/* "$OLDPWD/model/" 2>/dev/null || true
            echo -e "${GREEN}Model files copied to ./model/${NC}"
        fi
    else
        echo -e "${YELLOW}Warning: Failed to download model archive. You may need to download it manually.${NC}"
    fi
    
    cd "$OLDPWD"
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}Download completed!${NC}"
    echo "=========================================="
    echo ""
    echo "Note: Please update REPO_OWNER and REPO_NAME in this script if needed."
}

# 运行主函数
main
