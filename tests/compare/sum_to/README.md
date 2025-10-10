# Sum_to算子行为比较

## 概述
本文档比较了OriginDL、PyTorch和libtorch中sum_to算子的行为差异。

## 核心差异

### 1. 函数存在性
- **PyTorch**: 没有直接的`sum_to`函数
- **libtorch**: 有`torch::sum_to`函数
- **OriginDL**: 有`sum_to`函数

### 2. 广播行为
- **PyTorch**: 无直接对应，但可以通过`sum`+`expand`实现
- **libtorch**: `sum_to`**不支持广播**，当目标元素数量大于源元素数量时直接返回原始张量
- **OriginDL**: 与libtorch一致，**不支持广播**，当目标元素数量大于源元素数量时抛出异常

### 3. 错误处理
- **PyTorch**: 无直接对应
- **libtorch**: 静默返回原始张量
- **OriginDL**: 抛出`std::runtime_error`异常

## 测试用例

### 用例1: 正常压缩
```python
# PyTorch (通过sum+expand模拟)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])  # shape: [2, 3]
result = x.sum(dim=1).unsqueeze(1)  # shape: [2, 1]

# libtorch (C++)
auto x = torch::tensor({{1, 2, 3}, {4, 5, 6}});
auto result = torch::sum_to(x, {2, 1});  // shape: [2, 1]

# OriginDL
auto x = Tensor({1, 2, 3, 4, 5, 6}, Shape{2, 3});
auto result = sum_to(x, Shape{2, 1});  // shape: [2, 1]
```

### 用例2: 广播尝试（不支持）
```python
# PyTorch (通过expand实现)
x = torch.tensor([5.0])  # shape: [1]
result = x.expand(3)  # shape: [3], 结果: [5, 5, 5]

# libtorch (C++)
auto x = torch::tensor({5.0});
auto result = torch::sum_to(x, {3});  // 返回原始张量，shape: [1]

# OriginDL
auto x = Tensor({5.0}, Shape{1});
// 抛出异常: "sum_to: Target shape cannot have more elements than source tensor"
```

### 用例3: 标量压缩
```python
# PyTorch
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
result = x.sum()  # 标量: 21

# libtorch (C++)
auto x = torch::tensor({{1, 2, 3}, {4, 5, 6}});
auto result = torch::sum_to(x, {});  // 标量: 21

# OriginDL
auto x = Tensor({1, 2, 3, 4, 5, 6}, Shape{2, 3});
auto result = sum_to(x, Shape{});  // 标量: 21
```

## 关键发现

1. **libtorch的sum_to不支持广播**：这是与PyTorch的expand行为的重要差异
2. **OriginDL选择与libtorch保持一致**：不支持广播，并抛出异常以明确错误
3. **PyTorch没有直接的sum_to函数**：需要通过其他操作组合实现

## 结论
OriginDL的sum_to算子行为与libtorch完全一致，都不支持广播功能。当目标形状更大时，OriginDL抛出异常，而libtorch静默返回原始张量。这种设计选择确保了与libtorch后端的兼容性。
