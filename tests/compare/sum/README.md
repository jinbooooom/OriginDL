# Sum算子行为比较

## 概述
本文档比较了OriginDL、PyTorch和libtorch中sum算子的行为差异。

## 核心差异

### 1. 内存布局
- **PyTorch**: 行主序（row-major）
- **libtorch**: 行主序（row-major）
- **OriginDL**: 行主序（row-major，基于libtorch后端）

### 2. 全局求和行为
- **PyTorch**: `torch.sum(x)` 返回标量形状 `[]`
- **libtorch**: `torch.sum(x)` 返回标量形状 `[]`
- **OriginDL**: `sum(x, -1)` 返回形状 `[1]`（为了匹配测试期望）

### 3. 轴求和行为
- **PyTorch**: `torch.sum(x, dim=0, keepdim=False)` 压缩维度
- **libtorch**: `torch.sum(x, dim=0, keepdim=False)` 压缩维度
- **OriginDL**: `sum(x, 0)` 压缩维度（keepdim=False）

### 4. 数值结果
由于使用相同的行主序内存布局，三个框架的数值结果完全一致。

## 测试用例

### 用例1: 全局求和
```python
# PyTorch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = torch.sum(x)  # 21, shape: []

# libtorch (C++)
auto x = torch::tensor({{1, 2, 3}, {4, 5, 6}});
auto result = x.sum();  // 21, shape: []

# OriginDL
auto x = Tensor({1, 2, 3, 4, 5, 6}, Shape{2, 3});
auto result = sum(x, -1);  // 21, shape: [1]
```

### 用例2: 轴求和
```python
# PyTorch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = torch.sum(x, dim=0)  # [5, 7, 9], shape: [3]

# libtorch (C++)
auto x = torch::tensor({{1, 2, 3}, {4, 5, 6}});
auto result = x.sum(0, false);  // [5, 7, 9], shape: [3]

# OriginDL
auto x = Tensor({1, 2, 3, 4, 5, 6}, Shape{2, 3});
auto result = sum(x, 0);  // [5, 7, 9], shape: [3]
```

## 结论
OriginDL的sum算子行为与libtorch基本一致，唯一的差异是全局求和返回形状为`[1]`而不是`[]`，这是为了匹配现有测试的期望。
