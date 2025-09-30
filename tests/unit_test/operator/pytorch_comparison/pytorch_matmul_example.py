#!/usr/bin/env python3
"""
PyTorch MatMul操作示例
展示PyTorch的矩阵乘法操作行为
"""

import torch
import numpy as np

def pytorch_matmul_example():
    print("=== PyTorch MatMul操作示例 ===")
    
    # 基本矩阵乘法
    x = torch.tensor([[1.0, 2.0],
                      [3.0, 4.0]], dtype=torch.float32)
    w = torch.tensor([[5.0, 6.0],
                      [7.0, 8.0]], dtype=torch.float32)
    
    print(f"矩阵x形状: {x.shape}")
    print(f"矩阵x数据:\n{x}")
    print(f"矩阵w形状: {w.shape}")
    print(f"矩阵w数据:\n{w}")
    
    # 矩阵乘法
    result = torch.matmul(x, w)
    print(f"\n矩阵乘法结果形状: {result.shape}")
    print(f"矩阵乘法结果:\n{result}")
    
    # 使用@运算符
    result2 = x @ w
    print(f"\n使用@运算符结果:\n{result2}")
    
    # 不同尺寸的矩阵乘法
    print("\n=== 不同尺寸矩阵乘法 ===")
    x2 = torch.tensor([[1.0, 2.0, 3.0],
                       [4.0, 5.0, 6.0]], dtype=torch.float32)
    w2 = torch.tensor([[7.0, 8.0],
                       [9.0, 10.0],
                       [11.0, 12.0]], dtype=torch.float32)
    
    print(f"矩阵x2形状: {x2.shape}")
    print(f"矩阵x2数据:\n{x2}")
    print(f"矩阵w2形状: {w2.shape}")
    print(f"矩阵w2数据:\n{w2}")
    
    result3 = torch.matmul(x2, w2)
    print(f"\n不同尺寸矩阵乘法结果形状: {result3.shape}")
    print(f"不同尺寸矩阵乘法结果:\n{result3}")
    
    # 标量矩阵乘法
    print("\n=== 标量矩阵乘法 ===")
    x_scalar = torch.tensor([[2.0]], dtype=torch.float32)
    w_scalar = torch.tensor([[3.0]], dtype=torch.float32)
    
    print(f"标量矩阵x形状: {x_scalar.shape}")
    print(f"标量矩阵x数据: {x_scalar}")
    print(f"标量矩阵w形状: {w_scalar.shape}")
    print(f"标量矩阵w数据: {w_scalar}")
    
    result_scalar = torch.matmul(x_scalar, w_scalar)
    print(f"\n标量矩阵乘法结果形状: {result_scalar.shape}")
    print(f"标量矩阵乘法结果: {result_scalar}")

if __name__ == "__main__":
    pytorch_matmul_example()
