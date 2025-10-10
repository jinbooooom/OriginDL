#!/usr/bin/env python3
"""
PyTorch sum算子行为测试
"""
import torch

def test_pytorch_sum():
    print("=== PyTorch Sum算子测试 ===")
    
    # 测试1: 全局求和
    print("\n1. 全局求和测试:")
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    print(f"输入张量: {x}")
    print(f"输入形状: {x.shape}")
    
    result = torch.sum(x)
    print(f"全局求和结果: {result}")
    print(f"全局求和形状: {result.shape}")
    
    # 测试2: 轴求和
    print("\n2. 轴求和测试:")
    result0 = torch.sum(x, dim=0)
    print(f"沿轴0求和: {result0}")
    print(f"沿轴0求和形状: {result0.shape}")
    
    result1 = torch.sum(x, dim=1)
    print(f"沿轴1求和: {result1}")
    print(f"沿轴1求和形状: {result1.shape}")
    
    # 测试3: keepdim参数
    print("\n3. keepdim参数测试:")
    result0_keepdim = torch.sum(x, dim=0, keepdim=True)
    print(f"沿轴0求和(keepdim=True): {result0_keepdim}")
    print(f"沿轴0求和(keepdim=True)形状: {result0_keepdim.shape}")
    
    result0_no_keepdim = torch.sum(x, dim=0, keepdim=False)
    print(f"沿轴0求和(keepdim=False): {result0_no_keepdim}")
    print(f"沿轴0求和(keepdim=False)形状: {result0_no_keepdim.shape}")
    
    # 测试4: 三维张量
    print("\n4. 三维张量测试:")
    x3d = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    print(f"三维输入张量: {x3d}")
    print(f"三维输入形状: {x3d.shape}")
    
    result3d_0 = torch.sum(x3d, dim=0)
    print(f"沿轴0求和: {result3d_0}")
    print(f"沿轴0求和形状: {result3d_0.shape}")
    
    result3d_1 = torch.sum(x3d, dim=1)
    print(f"沿轴1求和: {result3d_1}")
    print(f"沿轴1求和形状: {result3d_1.shape}")
    
    result3d_2 = torch.sum(x3d, dim=2)
    print(f"沿轴2求和: {result3d_2}")
    print(f"沿轴2求和形状: {result3d_2.shape}")

if __name__ == "__main__":
    test_pytorch_sum()
