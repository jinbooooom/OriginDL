#!/usr/bin/env python3
"""
PyTorch Sum操作示例
展示PyTorch的sum操作在不同轴上的行为

重要说明：
- PyTorch默认行为：keepdim=False（压缩维度）
- OriginDL默认行为：keepdim=True（保持维度）
- 本示例展示了两种行为的差异
"""

import torch
import numpy as np

def pytorch_sum_example():
    print("=== PyTorch Sum操作示例 ===")
    
    # 创建测试数据
    x = torch.tensor([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]], dtype=torch.float32)
    print(f"原始张量形状: {x.shape}")
    print(f"原始张量数据:\n{x}")
    
    # 沿轴0求和（列求和）- 默认行为（压缩维度）
    sum_axis0 = torch.sum(x, dim=0)
    print(f"\n沿轴0求和结果形状（默认）: {sum_axis0.shape}")
    print(f"沿轴0求和结果（默认）: {sum_axis0}")
    
    # 沿轴0求和（列求和）- 保持维度
    sum_axis0_keepdim = torch.sum(x, dim=0, keepdim=True)
    print(f"\n沿轴0求和结果形状（keepdim=True）: {sum_axis0_keepdim.shape}")
    print(f"沿轴0求和结果（keepdim=True）: {sum_axis0_keepdim}")
    
    # 沿轴1求和（行求和）- 默认行为（压缩维度）
    sum_axis1 = torch.sum(x, dim=1)
    print(f"\n沿轴1求和结果形状（默认）: {sum_axis1.shape}")
    print(f"沿轴1求和结果（默认）: {sum_axis1}")
    
    # 沿轴1求和（行求和）- 保持维度
    sum_axis1_keepdim = torch.sum(x, dim=1, keepdim=True)
    print(f"\n沿轴1求和结果形状（keepdim=True）: {sum_axis1_keepdim.shape}")
    print(f"沿轴1求和结果（keepdim=True）: {sum_axis1_keepdim}")
    
    # 全局求和
    sum_all = torch.sum(x)
    print(f"\n全局求和结果形状: {sum_all.shape}")
    print(f"全局求和结果: {sum_all}")
    
    # 三维张量测试
    print("\n=== 三维张量测试 ===")
    x3d = torch.tensor([[[1.0, 2.0],
                         [3.0, 4.0]],
                        [[5.0, 6.0],
                         [7.0, 8.0]]], dtype=torch.float32)
    print(f"三维张量形状: {x3d.shape}")
    print(f"三维张量数据:\n{x3d}")
    
    # 沿轴0求和 - 默认行为（压缩维度）
    sum3d_axis0 = torch.sum(x3d, dim=0)
    print(f"\n三维张量沿轴0求和结果形状（默认）: {sum3d_axis0.shape}")
    print(f"三维张量沿轴0求和结果（默认）:\n{sum3d_axis0}")
    
    # 沿轴0求和 - 保持维度
    sum3d_axis0_keepdim = torch.sum(x3d, dim=0, keepdim=True)
    print(f"\n三维张量沿轴0求和结果形状（keepdim=True）: {sum3d_axis0_keepdim.shape}")
    print(f"三维张量沿轴0求和结果（keepdim=True）:\n{sum3d_axis0_keepdim}")

if __name__ == "__main__":
    pytorch_sum_example()
