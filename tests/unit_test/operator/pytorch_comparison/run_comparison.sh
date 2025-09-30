#!/bin/bash

echo "=== PyTorch与OriginDL行为对比示例 ==="
echo ""

# 检查Python是否可用
if command -v python3 &> /dev/null; then
    echo "开始成对对比示例..."
    echo ""
    
    # 1. Sum操作对比
    echo "=========================================="
    echo "1. Sum操作对比"
    echo "=========================================="
    echo ""
    echo "--- PyTorch Sum操作示例 ---"
    python3 pytorch_sum_example.py
    echo ""
    echo "--- OriginDL Sum操作示例 ---"
    if [ -f "../../../../build/bin/pytorch_comparison/origindl_sum_example" ]; then
        ../../../../build/bin/pytorch_comparison/origindl_sum_example
    else
        echo "OriginDL Sum示例未找到，请先运行: cd /datas/jinbo/s/ai && bash build.sh"
    fi
    echo ""
    
    # 2. Transpose操作对比
    echo "=========================================="
    echo "2. Transpose操作对比"
    echo "=========================================="
    echo ""
    echo "--- PyTorch Transpose操作示例 ---"
    python3 pytorch_transpose_example.py
    echo ""
    echo "--- OriginDL Transpose操作示例 ---"
    if [ -f "../../../../build/bin/pytorch_comparison/origindl_transpose_example" ]; then
        ../../../../build/bin/pytorch_comparison/origindl_transpose_example
    else
        echo "OriginDL Transpose示例未找到，请先运行: cd /datas/jinbo/s/ai && bash build.sh"
    fi
    echo ""
    
    # 3. MatMul操作对比
    echo "=========================================="
    echo "3. MatMul操作对比"
    echo "=========================================="
    echo ""
    echo "--- PyTorch MatMul操作示例 ---"
    python3 pytorch_matmul_example.py
    echo ""
    echo "--- OriginDL MatMul操作示例 ---"
    if [ -f "../../../../build/bin/pytorch_comparison/origindl_matmul_example" ]; then
        ../../../../build/bin/pytorch_comparison/origindl_matmul_example
    else
        echo "OriginDL MatMul示例未找到，请先运行: cd /datas/jinbo/s/ai && bash build.sh"
    fi
    echo ""
    
    # 4. Reshape操作对比
    echo "=========================================="
    echo "4. Reshape操作对比"
    echo "=========================================="
    echo ""
    echo "--- PyTorch Reshape操作示例 ---"
    python3 pytorch_reshape_example.py
    echo ""
    echo "--- OriginDL Reshape操作示例 ---"
    if [ -f "../../../../build/bin/pytorch_comparison/origindl_reshape_example" ]; then
        ../../../../build/bin/pytorch_comparison/origindl_reshape_example
    else
        echo "OriginDL Reshape示例未找到，请先运行: cd /datas/jinbo/s/ai && bash build.sh"
    fi
    echo ""
    
else
    echo "Python3未找到，跳过PyTorch示例"
    echo "请安装Python3来运行PyTorch对比示例"
fi

echo "=========================================="
echo "对比完成！"
