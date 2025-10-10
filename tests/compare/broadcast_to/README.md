# Broadcast_to算子行为比较

## 概述
本文档比较了OriginDL、PyTorch和libtorch中broadcast_to算子的行为差异。

## 核心差异

### 1. 函数名称
- **PyTorch**: `tensor.expand()`
- **libtorch**: `tensor.expand()`
- **OriginDL**: `broadcast_to()`

### 2. 广播规则
- **PyTorch**: 行主序，非单例维度必须匹配
- **libtorch**: 行主序，非单例维度必须匹配
- **OriginDL**: 行主序，非单例维度必须匹配（基于libtorch后端）

### 3. 内存布局
- **PyTorch**: 行主序（row-major）
- **libtorch**: 行主序（row-major）
- **OriginDL**: 行主序（row-major，基于libtorch后端）

## 测试用例

### 用例1: 基本广播
```python
# PyTorch
x = torch.tensor([1, 2])  # shape: [2]
result = x.expand(2, 2)   # shape: [2, 2], 结果: [[1,2],[1,2]]

# libtorch (C++)
auto x = torch::tensor({1, 2});
auto result = x.expand({2, 2});  // shape: [2, 2], 结果: [[1,2],[1,2]]

# OriginDL
auto x = Tensor({1, 2}, Shape{2});
auto result = broadcast_to(x, Shape{2, 2});  // shape: [2, 2], 结果: [1,2,1,2]
```

### 用例2: 标量广播
```python
# PyTorch
x = torch.tensor([5.0])   # shape: [1]
result = x.expand(3)      # shape: [3], 结果: [5, 5, 5]

# libtorch (C++)
auto x = torch::tensor({5.0});
auto result = x.expand({3});  // shape: [3], 结果: [5, 5, 5]

# OriginDL
auto x = Tensor({5.0}, Shape{1});
auto result = broadcast_to(x, Shape{3});  // shape: [3], 结果: [5, 5, 5]
```

### 用例3: 2D广播
```python
# PyTorch
x = torch.tensor([[1], [2]])  # shape: [2, 1]
result = x.expand(2, 3)       # shape: [2, 3], 结果: [[1,1,1],[2,2,2]]

# libtorch (C++)
auto x = torch::tensor({{1}, {2}});
auto result = x.expand({2, 3});  // shape: [2, 3], 结果: [[1,1,1],[2,2,2]]

# OriginDL
auto x = Tensor({1, 2}, Shape{2, 1});
auto result = broadcast_to(x, Shape{2, 3});  // shape: [2, 3], 结果: [1,1,1,2,2,2]
```

### 用例4: 无效广播（应该失败）
```python
# PyTorch
x = torch.tensor([1, 2, 3])  # shape: [3]
# x.expand(2, 2)  # 失败：非单例维度不匹配

# libtorch (C++)
auto x = torch::tensor({1, 2, 3});
// x.expand({2, 2});  // 失败：非单例维度不匹配

# OriginDL
auto x = Tensor({1, 2, 3}, Shape{3});
// broadcast_to(x, Shape{2, 2});  // 失败：非单例维度不匹配
```

## 关键发现

1. **广播规则一致**：三个框架都遵循相同的广播规则
2. **内存布局一致**：都使用行主序内存布局
3. **错误处理一致**：无效广播都会抛出异常
4. **数值结果一致**：由于使用相同的后端和内存布局，数值结果完全一致

## 注意事项

1. **维度匹配规则**：非单例维度必须完全匹配
2. **单例维度扩展**：单例维度（大小为1）可以扩展到任意大小
3. **内存效率**：expand操作是视图操作，不复制数据

## 结论
OriginDL的broadcast_to算子行为与libtorch的expand完全一致，都遵循相同的广播规则和内存布局。这确保了与libtorch后端的完全兼容性。
