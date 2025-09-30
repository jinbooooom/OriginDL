#!/usr/bin/env python3
"""
PyTorch Reshape操作示例
展示PyTorch的reshape操作行为
"""

import torch
import numpy as np

def pytorch_reshape_example():
    print("=== PyTorch Reshape操作示例 ===")
    
    # 基本reshape操作
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    print(f"原始张量形状: {x.shape}")
    print(f"原始张量数据: {x}")
    
    # reshape为4x1
    x_reshaped = x.reshape(4, 1)
    print(f"\nreshape为(4,1)后形状: {x_reshaped.shape}")
    print(f"reshape为(4,1)后数据:\n{x_reshaped}")
    
    # reshape为1x4
    x_reshaped2 = x.reshape(1, 4)
    print(f"\nreshape为(1,4)后形状: {x_reshaped2.shape}")
    print(f"reshape为(1,4)后数据: {x_reshaped2}")
    
    # 从2D到1D
    print("\n=== 从2D到1D ===")
    x2d = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0]], dtype=torch.float32)
    print(f"2D张量形状: {x2d.shape}")
    print(f"2D张量数据:\n{x2d}")
    
    x1d = x2d.reshape(-1)
    print(f"\nreshape为1D后形状: {x1d.shape}")
    print(f"reshape为1D后数据: {x1d}")
    
    # 从1D到2D
    print("\n=== 从1D到2D ===")
    x1d_orig = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    print(f"1D张量形状: {x1d_orig.shape}")
    print(f"1D张量数据: {x1d_orig}")
    
    x2d_new = x1d_orig.reshape(2, 3)
    print(f"\nreshape为(2,3)后形状: {x2d_new.shape}")
    print(f"reshape为(2,3)后数据:\n{x2d_new}")
    
    # 标量reshape
    print("\n=== 标量reshape ===")
    x_scalar = torch.tensor([5.0], dtype=torch.float32)
    print(f"标量张量形状: {x_scalar.shape}")
    print(f"标量张量数据: {x_scalar}")
    
    x_scalar_reshaped = x_scalar.reshape(1, 1)
    print(f"\n标量reshape为(1,1)后形状: {x_scalar_reshaped.shape}")
    print(f"标量reshape为(1,1)后数据: {x_scalar_reshaped}")

if __name__ == "__main__":
    pytorch_reshape_example()
