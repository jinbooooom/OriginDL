# MNIST 示例实现计划

## 概述

本文档详细说明了要实现类似 DeZero 的 `examples/mnist.py` (MNIST手写数字识别) 示例，OriginDL 需要实现的功能和实现步骤。

## 一、功能需求分析

### 1.1 DeZero MNIST 示例代码分析

```python
# 核心功能需求：
1. 数据集：MNIST(train=True/False)
2. 数据加载器：DataLoader(dataset, batch_size, shuffle)
3. 模型：MLP((hidden_size, hidden_size, 10), activation=F.relu)
4. 激活函数：F.relu
5. 损失函数：F.softmax_cross_entropy(y, t)
6. 评估函数：F.accuracy(y, t)
7. 优化器：Adam() + WeightDecay(1e-4) Hook
8. 配置管理：no_grad() 上下文管理器
9. GPU支持：to_gpu()
```

## 二、必需功能清单

### 2.1 激活函数（高优先级）

**需要实现：**
- ✅ `relu` - ReLU激活函数

**实现位置：**
- 头文件：`include/origin/core/operator.h`（所有算子类声明）
- 源文件：`src/operators/activation/relu.cpp`

**参考实现：**
```cpp
// ReLU: y = max(0, x)
// backward: gx = gy * (x > 0)
```

### 2.2 损失函数（高优先级）

**需要实现：**
- ✅ `softmax_cross_entropy` - Softmax交叉熵损失

**实现位置：**
- 头文件：`include/origin/core/operator.h`（所有算子类声明）
- 源文件：`src/operators/loss/softmax_cross_entropy.cpp`

**依赖：**
- `softmax` 函数（需要先实现）
- `log` 函数（需要先实现）

**参考实现：**
```cpp
// forward: 
//   1. 计算 softmax: p = exp(x - max(x)) / sum(exp(x - max(x)))
//   2. 计算交叉熵: loss = -mean(log(p[target]))
// backward:
//   gx = (softmax(x) - one_hot(target)) / N
```

### 2.3 评估函数（高优先级）

**需要实现：**
- ✅ `accuracy` - 准确率计算

**实现位置：**
- 头文件：`include/origin/utils/metrics.h` ⚠️ **不放在operators下**
- 源文件：`src/utils/metrics.cpp`

**设计说明：**
- ⚠️ **评估函数不应该放在 `operators/` 下**，因为：
  1. 评估函数（如accuracy）不需要反向传播，不是计算图的一部分
  2. `Operator` 基类要求实现 `backward()` 方法，但评估函数不需要
  3. 评估函数是工具函数，应该放在 `utils/` 下
- 返回标量张量（float类型）

**参考实现：**
```cpp
// accuracy = mean(argmax(y, axis=1) == t)
// 注意：这是一个不可微分的函数，不需要反向传播
```

### 2.4 Softmax 函数（高优先级）

**需要实现：**
- ✅ `softmax` - Softmax归一化

**实现位置：**
- 头文件：`include/origin/core/operator.h`
- 源文件：`src/operators/softmax.cpp`

**参考实现：**
```cpp
// softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
// 数值稳定版本：先减去最大值再计算
```

### 2.5 Log 函数（高优先级）

**需要实现：**
- ✅ `log` - 自然对数

**实现位置：**
- 头文件：`include/origin/core/operator.h`
- 源文件：`src/operators/log.cpp`

**参考实现：**
```cpp
// forward: y = log(x)
// backward: gx = gy / x
```

### 2.6 Adam 优化器（高优先级）

**需要实现：**
- ✅ `Adam` - Adam优化器

**实现位置：**
- 头文件：`include/origin/optim/adam.h`
- 源文件：`src/optim/adam.cpp`

**参考实现：**
```cpp
// Adam算法：
// m = beta1 * m + (1 - beta1) * grad
// v = beta2 * v + (1 - beta2) * grad^2
// m_hat = m / (1 - beta1^t)
// v_hat = v / (1 - beta2^t)
// param = param - lr * m_hat / (sqrt(v_hat) + eps)
```

### 2.7 WeightDecay Hook（高优先级）

**需要实现：**
- ✅ `WeightDecay` - 权重衰减Hook

**实现位置：**
- 头文件：`include/origin/optim/hooks.h`
- 源文件：`src/optim/hooks.cpp`

**参考实现：**
```cpp
// WeightDecay Hook:
// param.grad += rate * param.data
```

### 2.8 MLP 模型（高优先级）

**需要实现：**
- ✅ `MLP` - 多层感知机模型

**实现位置：**
- 头文件：`include/origin/nn/models/mlp.h`
- 源文件：`src/nn/models/mlp.cpp`

**参考实现：**
```cpp
class MLP : public Module {
    std::vector<Linear> layers_;
    std::function<Tensor(const Tensor&)> activation_;
    
    MLP(std::vector<int> hidden_sizes, 
        std::function<Tensor(const Tensor&)> activation);
    Tensor forward(const Tensor& x) override;
};
```

### 2.9 数据集基类和 MNIST 数据集（高优先级）

**需要实现：**
- ✅ `Dataset` - 数据集基类
- ✅ `MNIST` - MNIST数据集

**实现位置：**
- 头文件：`include/origin/data/dataset.h`, `include/origin/data/mnist.h`
- 源文件：`src/data/dataset.cpp`, `src/data/mnist.cpp`

**参考实现：**
```cpp
class Dataset {
    virtual std::pair<Tensor, Tensor> get_item(size_t index) = 0;
    virtual size_t size() const = 0;
};

class MNIST : public Dataset {
    // 下载和加载MNIST数据
    // 返回 (image, label) 对
};
```

### 2.10 DataLoader（高优先级）

**需要实现：**
- ✅ `DataLoader` - 数据加载器

**实现位置：**
- 头文件：`include/origin/data/dataloader.h`
- 源文件：`src/data/dataloader.cpp`

**参考实现：**
```cpp
class DataLoader {
    Dataset& dataset_;
    size_t batch_size_;
    bool shuffle_;
    
    // 迭代器接口
    std::pair<Tensor, Tensor> next();
};
```

### 2.11 配置管理（可选但推荐）

**需要实现：**
- ✅ `no_grad()` - 禁用梯度计算上下文管理器

**实现位置：**
- 头文件：`include/origin/core/config.h`
- 源文件：`src/core/config.cpp`

**设计说明：**
- **`no_grad()` 的作用**：
  1. 在推理（测试）时禁用梯度计算，节省内存和计算资源
  2. 避免在测试时构建计算图，提高推理速度
  3. 与 PyTorch 的 `torch.no_grad()` 行为一致

- **PyTorch 中的用法**：
```python
# PyTorch MNIST 示例
with torch.no_grad():
    for x, t in test_loader:
        y = model(x)
        loss = criterion(y, t)
        # 这里不需要计算梯度，节省内存
```

- **DeZero 中的用法**：
```python
# DeZero MNIST 示例
with dezero.no_grad():
    for x, t in test_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
```

- **OriginDL 中的用法（预期）**：
```cpp
// OriginDL MNIST 示例 - 使用通用的 no_grad() 函数
{
    auto guard = no_grad();  // 返回 RAII guard 对象
    for (auto [x, t] : test_loader) {
        auto y = model(x);
        auto loss = softmax_cross_entropy(y, t);
        // 这里不需要计算梯度
    }
}  // guard 析构时自动恢复
```

**实现优先级：**
- ⚠️ **中优先级**：MNIST demo 可以先不使用 `no_grad()`，但推荐实现
- 可以先在测试时手动不调用 `backward()`，后续再添加 `no_grad()` 支持

**参考实现（通用 no_grad() 函数）**：
```cpp
// 内部 RAII guard 类（不对外暴露）
class NoGradGuard {
    bool old_value_;
public:
    NoGradGuard();
    ~NoGradGuard();
    // 禁止拷贝，允许移动
    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;
    NoGradGuard(NoGradGuard&&) = default;
    NoGradGuard& operator=(NoGradGuard&&) = default;
};

// 全局配置
namespace Config {
    extern bool enable_backprop;
}

// 通用的 no_grad() 函数，返回 RAII guard 对象
// 与 PyTorch 和 DeZero 的用法一致
NoGradGuard no_grad() {
    return NoGradGuard();
}

// 使用方式（与 PyTorch/DeZero 一致）
{
    auto guard = no_grad();  // 禁用梯度计算
    // ... 推理代码 ...
}  // guard 析构时自动恢复
```

## 三、实现优先级和步骤

### 阶段一：核心算子（必需）

1. **Log 函数**
   - 实现 `log` 算子
   - 实现前向和反向传播
   - 放在 `src/operators/math/log.cpp`
   - 添加单元测试

2. **Softmax 函数**
   - 实现 `softmax` 算子
   - 注意数值稳定性（减去最大值）
   - 放在 `src/operators/math/softmax.cpp`
   - 添加单元测试

3. **ReLU 激活函数**
   - 实现 `relu` 算子
   - 实现前向和反向传播
   - 放在 `src/operators/activation/relu.cpp`
   - 添加单元测试

4. **SoftmaxCrossEntropy 损失函数**
   - 实现 `softmax_cross_entropy` 算子
   - 依赖 `softmax` 和 `log`
   - 放在 `src/operators/loss/softmax_cross_entropy.cpp`
   - 添加单元测试

5. **Accuracy 评估函数**
   - 实现 `accuracy` 函数（放在 `utils/metrics.cpp`，**不放在operators下**）
   - 不需要反向传播，不继承 `Operator`
   - 添加单元测试

### 阶段二：优化器和Hook（必需）

6. **Adam 优化器**
   - 实现 `Adam` 类
   - 支持学习率、beta1、beta2、eps参数
   - 添加单元测试

7. **WeightDecay Hook**
   - 实现 `WeightDecay` Hook类
   - 在优化器中注册Hook
   - 添加单元测试

### 阶段三：模型和数据处理（必需）

8. **MLP 模型**
   - 实现 `MLP` 类
   - 支持自定义隐藏层大小和激活函数
   - 添加单元测试

9. **Dataset 基类**
   - 实现 `Dataset` 抽象基类
   - 定义接口规范

10. **MNIST 数据集**
    - 实现 `MNIST` 数据集类
    - 支持下载和加载MNIST数据
    - 支持训练/测试集切换

11. **DataLoader**
    - 实现 `DataLoader` 类
    - 支持批处理、随机打乱
    - 实现迭代器接口

### 阶段四：配置和工具（可选，可延后）

12. **no_grad() 函数**（可选）
    - 实现全局配置管理
    - 实现通用的 `no_grad()` 函数，返回 RAII guard 对象
    - 与 PyTorch/DeZero 的用法保持一致
    - ⚠️ **可以先跳过**：MNIST demo 可以先在测试时不调用 `backward()`，后续再添加

## 四、实现细节

### 4.1 文件结构

```
include/origin/
├── core/
│   ├── operator.h          # 添加新算子声明（激活函数、数学计算、损失函数）
│   └── config.h           # 配置管理（新增，可选）
├── optim/
│   ├── adam.h             # Adam优化器（新增）
│   └── hooks.h            # Hook类（新增）
├── nn/
│   └── models/
│       └── mlp.h          # MLP模型（新增）
├── utils/
│   └── metrics.h          # 评估函数（新增，accuracy等）
└── data/                  # 数据处理模块（新增）
    ├── dataset.h
    ├── dataloader.h
    └── mnist.h

src/
├── operators/
│   ├── math/              # 数学运算和基础算子
│   │   ├── log.cpp        # 新增
│   │   ├── softmax.cpp    # 新增
│   │   └── ...            # 现有算子迁移到此目录
│   ├── activation/        # 激活函数
│   │   └── relu.cpp       # 新增
│   ├── loss/              # 损失函数
│   │   └── softmax_cross_entropy.cpp  # 新增
│   └── operator.cpp       # Operator 基类实现（保持在根目录）
│   # ⚠️ accuracy 不放在这里，放在 utils/metrics.cpp
├── optim/
│   ├── adam.cpp           # Adam优化器（新增）
│   └── hooks.cpp          # Hook实现（新增）
├── nn/
│   └── models/
│       └── mlp.cpp        # MLP模型（新增）
├── utils/
│   └── metrics.cpp        # 评估函数（新增，accuracy等）
└── data/                  # 数据处理模块（新增）
    ├── dataset.cpp
    ├── dataloader.cpp
    └── mnist.cpp
```

### 4.2 目录结构设计说明

**设计原则：**
1. **`src/operators/`** - 存放需要反向传播的算子
   - ✅ 激活函数（relu, sigmoid, tanh等）
   - ✅ 数学计算（log, exp, sin, cos等）
   - ✅ 损失函数（softmax_cross_entropy, mse_loss等）
   - ✅ 形状操作（reshape, transpose等）
   - ❌ **不包含**评估函数（accuracy等）

2. **`src/utils/metrics.cpp`** - 存放评估和指标函数
   - ✅ 准确率（accuracy）
   - ✅ 其他评估指标（precision, recall等，未来扩展）

3. **设计理由：**
   - `Operator` 基类要求实现 `forward()` 和 `backward()` 方法
   - 评估函数不需要反向传播，不应该继承 `Operator`
   - 评估函数是工具函数，放在 `utils/` 更合适
   - 与 PyTorch 设计一致：PyTorch 的 `accuracy` 也是工具函数，不是 `Function`

### 4.3 Operators 目录组织方案

**当前结构：**
```
src/operators/
├── add.cpp
├── sub.cpp
├── mul.cpp
├── div.cpp
├── exp.cpp
├── pow.cpp
├── square.cpp
├── reshape.cpp
├── transpose.cpp
├── sum.cpp
├── mat_mul.cpp
├── neg.cpp
├── broadcast_to.cpp
├── sum_to.cpp
└── operator.cpp
```

**推荐方案：使用三个子目录分类组织**

```
src/operators/
├── math/                    # 数学运算和基础算子
│   ├── add.cpp
│   ├── sub.cpp
│   ├── mul.cpp
│   ├── div.cpp
│   ├── neg.cpp
│   ├── exp.cpp
│   ├── pow.cpp
│   ├── square.cpp
│   ├── log.cpp              # 新增
│   ├── softmax.cpp          # 新增
│   ├── reshape.cpp
│   ├── transpose.cpp
│   ├── sum.cpp
│   ├── broadcast_to.cpp
│   ├── sum_to.cpp
│   └── mat_mul.cpp
├── activation/              # 激活函数
│   ├── relu.cpp             # 新增
│   └── ...                  # 未来：sigmoid, tanh 等
├── loss/                    # 损失函数
│   ├── softmax_cross_entropy.cpp  # 新增
│   └── ...                  # 未来：mse_loss, cross_entropy 等
└── operator.cpp             # Operator 基类实现（保持在根目录）
```

**方案优势：**
1. ✅ **清晰的分类**：按功能分类，易于查找和维护
2. ✅ **易于扩展**：未来添加新的激活函数或损失函数时，直接放入对应目录
3. ✅ **符合直觉**：目录结构直观，新成员容易理解
4. ✅ **规模适中**：三个目录不会过于复杂，也不会过于简单
5. ✅ **与 PyTorch 思路一致**：参考 PyTorch 的分类思想，但更简洁

**文件迁移计划：**

**需要迁移到 `math/` 的文件：**
- `add.cpp` → `math/add.cpp`
- `sub.cpp` → `math/sub.cpp`
- `mul.cpp` → `math/mul.cpp`
- `div.cpp` → `math/div.cpp`
- `neg.cpp` → `math/neg.cpp`
- `exp.cpp` → `math/exp.cpp`
- `pow.cpp` → `math/pow.cpp`
- `square.cpp` → `math/square.cpp`
- `reshape.cpp` → `math/reshape.cpp`
- `transpose.cpp` → `math/transpose.cpp`
- `sum.cpp` → `math/sum.cpp`
- `broadcast_to.cpp` → `math/broadcast_to.cpp`
- `sum_to.cpp` → `math/sum_to.cpp`
- `mat_mul.cpp` → `math/mat_mul.cpp`

**保持不变：**
- `operator.cpp` - 保持在 `src/operators/` 根目录

**新增文件：**
- `math/log.cpp` - Log 算子
- `math/softmax.cpp` - Softmax 算子
- `activation/relu.cpp` - ReLU 激活函数
- `loss/softmax_cross_entropy.cpp` - Softmax交叉熵损失函数

**头文件组织：**
- 所有算子类声明仍然在 `include/origin/core/operator.h`
- 保持统一接口，便于使用

### 4.4 底层 Mat 算子的目录组织

**当前结构：**
```
src/mat/origin/
├── cpu/
│   ├── add.cpp
│   ├── subtract.cpp
│   ├── multiply.cpp
│   ├── divide.cpp
│   ├── exp.cpp
│   ├── log.cpp
│   ├── pow.cpp
│   ├── square.cpp
│   ├── sqrt.cpp
│   ├── reshape.cpp
│   ├── transpose.cpp
│   ├── sum.cpp
│   ├── matmul.cpp
│   └── ...
└── cuda/
    ├── add.cu
    ├── subtract.cu
    ├── multiply.cu
    ├── divide.cu
    ├── exp.cu
    ├── log.cu
    ├── pow.cu
    ├── square.cu
    ├── sqrt.cu
    ├── reshape.cu
    ├── transpose.cu
    ├── sum.cu
    ├── mat_mul.cu
    └── ...
```

**推荐方案：与 operators/ 保持一致，使用三个子目录**

```
src/mat/origin/
├── cpu/
│   ├── math/                    # 数学运算和基础算子
│   │   ├── add.cpp
│   │   ├── subtract.cpp
│   │   ├── multiply.cpp
│   │   ├── divide.cpp
│   │   ├── negate.cpp
│   │   ├── exp.cpp
│   │   ├── log.cpp
│   │   ├── pow.cpp
│   │   ├── square.cpp
│   │   ├── sqrt.cpp
│   │   ├── reshape.cpp
│   │   ├── transpose.cpp
│   │   ├── sum.cpp
│   │   ├── sum_to.cpp
│   │   ├── broadcast_to.cpp
│   │   └── matmul.cpp
│   ├── activation/              # 激活函数（未来）
│   │   └── relu.cpp             # 新增
│   ├── loss/                    # 损失函数（未来）
│   │   └── softmax_cross_entropy.cpp  # 新增
│   ├── cpu_kernels.cpp         # CPU 内核实现（保持在根目录）
│   ├── factory.cpp              # 工厂函数（保持在根目录）
│   └── type_conversion.cpp      # 类型转换（保持在根目录）
└── cuda/
    ├── math/                    # 数学运算和基础算子
    │   ├── add.cu
    │   ├── subtract.cu
    │   ├── multiply.cu
    │   ├── divide.cu
    │   ├── negate.cu
    │   ├── exp.cu
    │   ├── log.cu
    │   ├── pow.cu
    │   ├── square.cu
    │   ├── sqrt.cu
    │   ├── reshape.cu
    │   ├── transpose.cu
    │   ├── sum.cu
    │   ├── sum_to.cu
    │   ├── broadcast_to.cu
    │   └── mat_mul.cu
    ├── activation/              # 激活函数（未来）
    │   └── relu.cu              # 新增
    ├── loss/                    # 损失函数（未来）
    │   └── softmax_cross_entropy.cu  # 新增
    ├── cuda_kernels.cu          # CUDA 内核实现（保持在根目录）
    ├── cuda_utils.cu            # CUDA 工具函数（保持在根目录）
    ├── stream_manager.cu        # 流管理（保持在根目录）
    └── factory.cu               # 工厂函数（保持在根目录）
```

**方案优势：**
1. ✅ **与 operators/ 保持一致**：上层算子和底层实现使用相同的目录结构
2. ✅ **易于对应**：`operators/math/add.cpp` 对应 `mat/origin/cpu/math/add.cpp`
3. ✅ **清晰的分类**：按功能分类，便于查找和维护
4. ✅ **易于扩展**：未来添加新的激活函数或损失函数时，直接放入对应目录

**文件迁移计划：**

**CPU 算子迁移到 `cpu/math/`：**
- `add.cpp` → `math/add.cpp`
- `subtract.cpp` → `math/subtract.cpp`
- `multiply.cpp` → `math/multiply.cpp`
- `divide.cpp` → `math/divide.cpp`
- `negate.cpp` → `math/negate.cpp`
- `exp.cpp` → `math/exp.cpp`
- `log.cpp` → `math/log.cpp`
- `pow.cpp` → `math/pow.cpp`
- `square.cpp` → `math/square.cpp`
- `sqrt.cpp` → `math/sqrt.cpp`
- `reshape.cpp` → `math/reshape.cpp`
- `transpose.cpp` → `math/transpose.cpp`
- `sum.cpp` → `math/sum.cpp`
- `sum_to.cpp` → `math/sum_to.cpp`
- `broadcast_to.cpp` → `math/broadcast_to.cpp`（如果存在）
- `matmul.cpp` → `math/matmul.cpp`

**CUDA 算子迁移到 `cuda/math/`：**
- 对应的 `.cu` 文件也迁移到 `cuda/math/` 目录

**保持不变：**
- `cpu_kernels.cpp` / `cuda_kernels.cu` - 内核实现
- `factory.cpp` / `factory.cu` - 工厂函数
- `type_conversion.cpp` / `type_conversion.cu` - 类型转换
- `cuda_utils.cu` - CUDA 工具函数
- `stream_manager.cu` - 流管理

**头文件组织：**
- `include/origin/mat/origin/cpu/cpu_ops.h` - 可以保持统一，或按分类拆分
- `include/origin/mat/origin/cuda/cuda_ops.cuh` - 可以保持统一，或按分类拆分
- 建议先保持统一头文件，未来再考虑拆分

### 4.5 CMakeLists.txt 更新

需要在 `CMakeLists.txt` 中更新算子源文件的收集方式：

**当前（需要修改）：**
```cmake
file(GLOB MY_LIB_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp
)
```

**更新后（支持子目录）：**
```cmake
# 算子源文件（支持子目录）
file(GLOB OPERATOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/math/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/activation/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/loss/*.cpp
)

# 或者使用递归 GLOB（更简洁）
file(GLOB_RECURSE OPERATOR_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/operators/*.cpp
)

# 其他新增模块
file(GLOB OPTIM_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/optim/*.cpp
)

file(GLOB MODEL_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/nn/models/*.cpp
)

file(GLOB DATA_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/data/*.cpp
)
```

**注意：**
- 使用 `GLOB_RECURSE` 可以自动递归收集所有子目录的 `.cpp` 文件
- 但需要确保 `operator.cpp` 在根目录，不会被重复包含

**CPU/CUDA 算子的 CMakeLists.txt 更新：**

**当前（需要修改）：**
```cmake
file(GLOB ORIGIN_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/*.cpp)
```

**更新后（支持子目录）：**
```cmake
# CPU 算子源文件（支持子目录）
file(GLOB_RECURSE ORIGIN_CPU_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/*.cpp
)

# CUDA 算子源文件（支持子目录）
file(GLOB_RECURSE ORIGIN_CUDA_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cuda/*.cu
)
```

**或者显式列出（更精确）：**
```cmake
# CPU 算子源文件
file(GLOB ORIGIN_CPU_MATH_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/math/*.cpp
)
file(GLOB ORIGIN_CPU_ACTIVATION_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/activation/*.cpp
)
file(GLOB ORIGIN_CPU_LOSS_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/loss/*.cpp
)
file(GLOB ORIGIN_CPU_OTHER_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cpu/*.cpp  # 根目录文件
)
list(APPEND ORIGIN_CPU_SRCS
    ${ORIGIN_CPU_MATH_SRCS}
    ${ORIGIN_CPU_ACTIVATION_SRCS}
    ${ORIGIN_CPU_LOSS_SRCS}
    ${ORIGIN_CPU_OTHER_SRCS}
)
```

### 4.3 示例代码结构

MNIST 示例代码应该放在：
```
tests/example/mnist/
├── CMakeLists.txt
└── mnist.cpp
```

## 五、测试计划

### 5.1 单元测试

每个新功能都需要添加单元测试：

1. **算子测试**：`tests/operators/test_*.cpp`
2. **优化器测试**：`tests/optim/test_*.cpp`
3. **模型测试**：`tests/nn/test_*.cpp`
4. **数据处理测试**：`tests/data/test_*.cpp`

### 5.2 集成测试

MNIST 示例本身就是一个集成测试，需要：
- 能够成功训练
- 训练损失下降
- 测试准确率提升
- 与 PyTorch 结果对比验证

## 六、参考实现

### 6.1 DeZero 参考

- `dezero/functions.py` - 所有算子的实现
- `dezero/optimizers.py` - 优化器实现
- `dezero/models.py` - MLP模型实现
- `dezero/datasets.py` - 数据集实现
- `dezero/dataloaders.py` - 数据加载器实现

### 6.2 PyTorch 参考

- PyTorch 的 C++ API 可以作为实现参考
- 注意数值稳定性和性能优化

## 七、设计原则和注意事项

### 7.1 设计原则

1. **参考 DeZero 设计，行为与 PyTorch 保持一致**
   - 设计思路参考 DeZero（教育性、简洁性）
   - 但行为要与 PyTorch 保持一致（数值结果、API 风格）

2. **目录结构设计**
   - `operators/` - 只包含需要反向传播的算子
   - `utils/metrics.cpp` - 评估函数（不需要反向传播）
   - 保持当前架构，最小化更改

3. **最快实现策略**
   - 优先实现必需功能，可选功能可延后
   - 先实现 CPU 版本，GPU 支持可后续添加
   - `no_grad()` 可以先跳过，测试时不调用 `backward()` 即可

### 7.2 注意事项

1. **数值稳定性**
   - Softmax 需要减去最大值避免溢出
   - Log 需要处理接近0的值

2. **内存管理**
   - DataLoader 需要高效的内存管理
   - 避免不必要的数据拷贝

3. **GPU 支持**
   - 所有新算子都需要支持 CUDA 后端
   - 可以先实现 CPU 版本，再添加 GPU 支持

4. **代码风格**
   - 遵循项目的代码风格规范
   - 参考现有代码的实现模式

5. **文档**
   - 为新功能添加详细的注释
   - 更新 API 文档

6. **测试验证**
   - 每个新功能都要与 PyTorch 结果对比验证
   - 确保数值结果一致（允许小的浮点误差）

## 八、最快实现路径（最小更改）

### 8.1 最小必需功能清单

为了最快完成 MNIST demo，以下是最小必需功能：

**必需（11项）：**
1. ✅ Log 算子
2. ✅ Softmax 算子
3. ✅ ReLU 激活函数
4. ✅ SoftmaxCrossEntropy 损失函数
5. ✅ Accuracy 评估函数（放在 utils/metrics.cpp）
6. ✅ Adam 优化器
7. ✅ WeightDecay Hook
8. ✅ MLP 模型
9. ✅ Dataset 基类
10. ✅ MNIST 数据集
11. ✅ DataLoader

**可选（可延后）：**
- ❌ `no_grad()` - 可以先跳过，测试时不调用 `backward()` 即可
- ❌ GPU 支持 - 可以先只实现 CPU 版本

### 8.2 实现顺序（最快路径）

**第一周：核心算子（5项）**
1. Log 算子（1天）
2. Softmax 算子（1天）
3. ReLU 激活函数（1天）
4. SoftmaxCrossEntropy 损失函数（1天）
5. Accuracy 评估函数（0.5天，放在 utils/）

**第二周：优化器和模型（3项）**
6. Adam 优化器（2天）
7. WeightDecay Hook（0.5天）
8. MLP 模型（1.5天）

**第三周：数据处理（3项）**
9. Dataset 基类（1天）
10. MNIST 数据集（2天）
11. DataLoader（1天）

**第四周：集成和测试**
12. MNIST demo 集成（2天）
13. 测试和调试（3天）

**总计**：约 15-18 天（最快路径）

### 8.3 最小更改原则

1. **保持当前架构**
   - 不改变现有的目录结构
   - 不改变现有的接口设计
   - 只添加新功能，不修改现有代码

2. **最小依赖**
   - 先实现 CPU 版本
   - 可选功能（如 no_grad）可延后

3. **快速验证**
   - 每个功能实现后立即测试
   - 与 PyTorch 结果对比验证

## 九、完成标准

MNIST 示例能够成功运行，并且：
1. 训练损失能够正常下降
2. 测试准确率能够达到 90% 以上（5个epoch）
3. 代码通过所有单元测试
4. 与 PyTorch 结果对比误差在可接受范围内

