#!/usr/bin/env python3
"""
PyTorch对比测试用例
与tests/originDL/operator/main.cpp保持完全一致的测试数据与用例
用于验证OriginDL的自动求导实现是否正确
"""

import torch
import math

def test_operators():
    """测试所有算子并与OriginDL结果对比"""
    
    print("=" * 80)
    print("PyTorch对比测试 - 验证OriginDL自动求导正确性")
    print("=" * 80)
    
    # 与C++版本完全一致的测试数据
    val0 = 2.0
    val1 = 4.0
    dim = (2, 2)
    
    # 创建与C++版本相同的张量
    x0 = torch.full(dim, val0, requires_grad=True)
    x1 = torch.full(dim, val1, requires_grad=True)
    
    print(f"测试数据: val0={val0}, val1={val1}, dim={dim}")
    print(f"x0 shape: {x0.shape}, x1 shape: {x1.shape}")
    print()
    
    # 1. 测试Neg算子: y = -x0
    print("1. Neg算子测试: y = -x0")
    x0.grad = None
    y = -x0
    y.backward(torch.ones_like(y))
    print(f"y = -x0:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print()
    
    # 2. 测试Add算子: y = x0 + x1
    print("2. Add算子测试: y = x0 + x1")
    x0.grad = None
    x1.grad = None
    y = x0 + x1
    y.backward(torch.ones_like(y))
    print(f"y = x0 + x1:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print(f"  dx1: {x1.grad.numpy()}")
    print()
    
    # 3. 测试Sub算子: y = x0 - x1
    print("3. Sub算子测试: y = x0 - x1")
    x0.grad = None
    x1.grad = None
    y = x0 - x1
    y.backward(torch.ones_like(y))
    print(f"y = x0 - x1:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print(f"  dx1: {x1.grad.numpy()}")
    print()
    
    # 4. 测试Mul算子: y = x0 * x1
    print("4. Mul算子测试: y = x0 * x1")
    x0.grad = None
    x1.grad = None
    y = x0 * x1
    y.backward(torch.ones_like(y))
    print(f"y = x0 * x1:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print(f"  dx1: {x1.grad.numpy()}")
    print()
    
    # 5. 测试Div算子: y = x0 / x1
    print("5. Div算子测试: y = x0 / x1")
    x0.grad = None
    x1.grad = None
    y = x0 / x1
    y.backward(torch.ones_like(y))
    print(f"y = x0 / x1:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print(f"  dx1: {x1.grad.numpy()}")
    print()
    
    # 6. 测试Square算子: y = x0^2
    print("6. Square算子测试: y = x0^2")
    x0.grad = None
    y = x0 ** 2
    y.backward(torch.ones_like(y))
    print(f"y = x0^2:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print()
    
    # 7. 测试Pow算子: y = x0^3
    print("7. Pow算子测试: y = x0^3")
    x0.grad = None
    y = x0 ** 3
    y.backward(torch.ones_like(y))
    print(f"y = x0^3:")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print()
    
    # 8. 测试Exp算子: y = exp(x0)
    print("8. Exp算子测试: y = exp(x0)")
    x0.grad = None
    y = torch.exp(x0)
    y.backward(torch.ones_like(y))
    print(f"y = exp(x0):")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx0: {x0.grad.numpy()}")
    print()
    
    # 9. 测试Reshape算子: y = reshape(x, {4, 2})
    print("9. Reshape算子测试: y = reshape(x, {4, 2})")
    # 与C++版本完全一致的数据: {0,1,2,3,4,5,6,7}, Shape{2, 4}
    x = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=torch.float32, requires_grad=True)
    print(f"  x: {x.detach().numpy()}")
    y = x.reshape(4, 2)
    y.backward(torch.ones_like(y))
    print(f"y = reshape(x, (4, 2)):")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx: {x.grad.numpy()}")
    print()
    
    # 10. 测试Transpose算子: y = transpose(x)
    print("10. Transpose算子测试: y = transpose(x)")
    x.grad = None
    y = x.t()
    y.backward(torch.ones_like(y))
    print(f"y = transpose(x):")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx: {x.grad.numpy()}")
    print()
    
    # 11. 测试Sum算子: y = sum(x)
    print("11. Sum算子测试: y = sum(x)")
    x.grad = None
    y = x.sum(dim=0, keepdim=True)  # 按行求和，与C++版本一致
    y.backward(torch.ones_like(y))
    print(f"y = sum(x, dim=0):")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx: {x.grad.numpy()}")
    print()
    
    # 12. 测试BroadcastTo算子: y = broadcast_to(x, {2, 2, 4})
    print("12. BroadcastTo算子测试: y = broadcast_to(x, {2, 2, 4})")
    x.grad = None
    # 通过expand实现广播到(2, 2, 4)
    y = x.unsqueeze(0).expand(2, 2, 4)
    y.backward(torch.ones_like(y))
    print(f"y = broadcast_to(x, (2, 2, 4)):")
    print(f"  y shape: {y.shape}")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx: {x.grad.numpy()}")
    print()
    
    # 13. 测试SumTo算子: y = sum_to(x, {1, 1})
    print("13. SumTo算子测试: y = sum_to(x, {1, 1})")
    x.grad = None
    y = x.sum().reshape(1, 1)  # 所有元素求和并reshape到(1,1)
    y.backward(torch.ones_like(y))
    print(f"y = sum_to(x, (1, 1)):")
    print(f"  y: {y.detach().numpy()}")
    print(f"  dx: {x.grad.numpy()}")
    print()
    
    # 14. 测试MatMul算子: y = mat_mul(a, b)
    print("14. MatMul算子测试: y = mat_mul(a, b)")
    # 与C++版本一致的2x2矩阵
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True)
    b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32, requires_grad=True)
    print(f"  a: {a.detach().numpy()}")
    print(f"  b: {b.detach().numpy()}")
    y = torch.mm(a, b)
    y.backward(torch.ones_like(y))
    print(f"y = mat_mul(a, b):")
    print(f"  y: {y.detach().numpy()}")
    print(f"  da: {a.grad.numpy()}")
    print(f"  db: {b.grad.numpy()}")
    print()
    
    print("=" * 80)
    print("PyTorch对比测试完成")
    print("请对比上述结果与OriginDL C++版本的输出")
    print("=" * 80)

if __name__ == "__main__":
    test_operators()