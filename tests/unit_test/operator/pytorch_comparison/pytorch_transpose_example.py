#!/usr/bin/env python3
"""
PyTorch Transpose操作示例
展示PyTorch的transpose操作行为
"""

import torch
import numpy as np

def pytorch_transpose_example():
    print("=== PyTorch Transpose操作示例 ===")
    
    # 2D矩阵转置
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=torch.float32)
    print(f"原始矩阵形状: {x.shape}")
    print(f"原始矩阵数据:\n{x}")
    
    # 转置操作
    x_t = torch.transpose(x, 0, 1)
    print(f"\n转置后形状: {x_t.shape}")
    print(f"转置后数据:\n{x_t}")
    
    # 一维张量转置
    print("\n=== 一维张量转置 ===")
    x1d = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    print(f"一维张量形状: {x1d.shape}")
    print(f"一维张量数据: {x1d}")
    
    # 一维张量转置（应该保持一维）
    x1d_t = torch.transpose(x1d, 0, 0)
    print(f"\n一维张量转置后形状: {x1d_t.shape}")
    print(f"一维张量转置后数据: {x1d_t}")
    
    # 方阵转置
    print("\n=== 方阵转置 ===")
    x_square = torch.tensor([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]], dtype=torch.float32)
    print(f"方阵形状: {x_square.shape}")
    print(f"方阵数据:\n{x_square}")
    
    x_square_t = torch.transpose(x_square, 0, 1)
    print(f"\n方阵转置后形状: {x_square_t.shape}")
    print(f"方阵转置后数据:\n{x_square_t}")

if __name__ == "__main__":
    pytorch_transpose_example()
