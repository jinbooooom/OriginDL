# PyTorch与OriginDL行为对比示例

本目录包含了PyTorch和OriginDL在相同操作上的行为对比示例，用于分析两者之间的差异。

## 文件结构

```
pytorch_comparison/
├── README.md                           # 本说明文件
├── pytorch_sum_example.py             # PyTorch sum操作示例
├── origindl_sum_example.cpp           # OriginDL sum操作示例
├── pytorch_transpose_example.py        # PyTorch transpose操作示例
├── origindl_transpose_example.cpp      # OriginDL transpose操作示例
├── pytorch_matmul_example.py          # PyTorch matmul操作示例
├── origindl_matmul_example.cpp         # OriginDL matmul操作示例
├── pytorch_reshape_example.py         # PyTorch reshape操作示例
└── origindl_reshape_example.cpp        # OriginDL reshape操作示例
```

## 使用方法

### 运行PyTorch示例

```bash
# 运行sum操作对比
python pytorch_sum_example.py

# 运行transpose操作对比
python pytorch_transpose_example.py

# 运行matmul操作对比
python pytorch_matmul_example.py

# 运行reshape操作对比
python pytorch_reshape_example.py
```

## 主要差异分析

### 核心差异：内存布局
- **PyTorch**: 使用行主序（row-major）内存布局
- **OriginDL**: 使用列主序（column-major）内存布局（ArrayFire特性）
- **影响**: 这会导致数据存储、访问和计算结果的根本性差异

### 1. Sum操作差异
- **PyTorch**: 默认 `keepdim=False`，自动压缩求和轴
- **OriginDL**: 默认 `keepdim=True`，保持维度结构
- **关键差异**: 
  - PyTorch: `torch.sum(x, dim=0)` 返回 `[3]` 形状
  - OriginDL: `sum(x, 0)` 返回 `[1,3]` 形状（保持维度）
  - 要匹配PyTorch行为，OriginDL需要手动压缩维度
- **列主序影响**: 由于内存布局不同，求和结果的数据排列可能不同

### 2. Transpose操作差异
- **PyTorch**: 一维张量转置后仍是一维
- **OriginDL**: 一维张量转置后可能变成二维
- **列主序影响**: 转置操作在列主序和行主序下的行为完全不同

### 3. MatMul操作差异
- **PyTorch**: 使用row-major内存布局
- **OriginDL**: 使用column-major内存布局，导致结果不同
- **列主序影响**: 矩阵乘法的结果会因为内存布局差异而完全不同

### 4. Reshape操作差异
- **PyTorch**: 精确保持目标形状
- **OriginDL**: 可能自动压缩某些维度
- **列主序影响**: reshape后的数据排列顺序可能不同

## 注意事项

1. 运行OriginDL示例前需要确保ArrayFire库已正确安装
2. 编译时需要指定正确的头文件路径和库路径
3. 这些示例主要用于分析行为差异，不是完整的测试用例
4. 如果遇到编译错误，请检查ArrayFire的安装路径和版本

## 扩展

可以基于这些示例添加更多的操作对比，如：
- broadcast_to操作
- sum_to操作
- 其他数学运算操作

每个操作都应该包含PyTorch和OriginDL的对应实现，以便进行详细的行为对比分析。
