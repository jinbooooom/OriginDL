# 1. 架构总览与设计理念

## 1.1 OriginDL 架构概览

OriginDL 是一个从零开始构建的深度学习框架，采用 C++ 实现，支持自动求导和多种计算后端。框架采用分层架构设计和模块化组织，将系统划分为多个清晰的层次和功能模块，每一层都有明确的职责，通过抽象接口实现层间解耦。

OriginDL 的整体架构由以下几个核心模块组成：

```mermaid
graph TB
    subgraph "应用层 (Application Layer)"
        UserCode["用户代码<br/>模型定义、训练、推理"]
    end
    
    subgraph "API层 (API Layer)"
        TensorAPI["Tensor API<br/>张量操作、运算符重载"]
        NNModule["Neural Network<br/>Module, Layer, Sequential"]
        Optim["Optimizer<br/>SGD, Adam"]
        DataModule["Data<br/>Dataset, DataLoader"]
        IOModule["IO<br/>Checkpoint, Model IO"]
        PNNXModule["PNNX<br/>模型推理"]
    end
    
    subgraph "核心计算层 (Core Computing Layer)"
        TensorImpl["TensorImpl<br/>数据管理、计算图管理"]
        Operator["Operator<br/>算子系统<br/>数学、激活、卷积等"]
        Autograd["自动求导系统<br/>反向传播、梯度计算"]
    end
    
    subgraph "抽象接口层 (Abstraction Layer)"
        MatInterface["Mat<br/>矩阵计算抽象接口<br/>统一多后端接口"]
    end
    
    subgraph "后端实现层 (Backend Implementation Layer)"
        OriginMat["OriginMat<br/>CPU/CUDA实现<br/>自研后端"]
        TorchMat["TorchMat<br/>LibTorch实现<br/>高度优化"]
        Storage["Storage<br/>内存管理"]
    end
    
    UserCode --> TensorAPI
    UserCode --> NNModule
    UserCode --> Optim
    UserCode --> DataModule
    UserCode --> IOModule
    UserCode --> PNNXModule
    
    TensorAPI --> TensorImpl
    NNModule --> TensorAPI
    NNModule --> Operator
    Optim --> TensorAPI
    Operator --> TensorImpl
    Operator --> MatInterface
    TensorImpl --> Autograd
    TensorImpl --> MatInterface
    Autograd --> Operator
    
    MatInterface -.->|继承实现| OriginMat
    MatInterface -.->|继承实现| TorchMat
    OriginMat --> Storage
    TorchMat --> Storage
    
    style UserCode fill:#e1f5ff
    style TensorAPI fill:#fff4e1
    style NNModule fill:#fff4e1
    style Optim fill:#fff4e1
    style DataModule fill:#fff4e1
    style IOModule fill:#fff4e1
    style PNNXModule fill:#fff4e1
    style TensorImpl fill:#ffe1f5
    style Operator fill:#ffe1f5
    style Autograd fill:#ffe1f5
    style MatInterface fill:#e1ffe1
    style OriginMat fill:#f5e1ff
    style TorchMat fill:#f5e1ff
    style Storage fill:#f5f5f5
```

**架构层级说明：**

1. **应用层**：用户编写的模型定义、训练和推理代码
2. **API层**：提供面向用户的高级接口，包括张量操作、神经网络模块、优化器、数据处理、模型IO和推理功能
3. **核心计算层**：实现框架的核心功能，包括张量实现、算子系统、自动求导机制
4. **抽象接口层**：定义统一的矩阵计算接口，实现后端解耦
5. **后端实现层**：提供具体的计算后端实现和内存管理

**Tensor 系统**是核心计算层的基础，它采用了四层架构设计（Tensor → TensorImpl → Mat → 后端实现），体现了分层架构的核心思想：**职责分离**和**依赖倒置**。每一层只依赖下层提供的接口，不依赖具体实现，从而实现解耦。详细的 Tensor 系统架构设计请参见 [Tensor 系统架构](#tensor-系统架构) 章节。

OriginDL 框架遵循以下核心设计原则：

1. **分层架构**：采用清晰的分层设计，每一层都有明确的职责，通过抽象接口实现层间解耦
2. **职责分离**：每个模块只负责自己的核心功能，避免职责混乱
3. **依赖倒置**：上层依赖下层的抽象接口，不依赖具体实现，支持多后端切换
4. **值语义设计**：Tensor 采用值语义，提供直观的用户体验
5. **自动求导**：支持动态计算图，自动构建和反向传播

# 2. Tensor 系统架构

## 2.1 Tensor 分层架构设计

Tensor 系统采用四层架构设计，从用户接口到数据存储，每一层都有明确的职责和清晰的边界。

### 2.1.1 四层架构概览

**架构层次图：**

```mermaid
flowchart TB
    subgraph UserLayer["用户代码层 (User Code)"]
        UserCode["Tensor y = x0 + x1;<br/>y.backward();"]
    end
    
    subgraph Layer1["第1层：Tensor (用户接口层)"]
        Tensor["Tensor<br/>- 值语义包装<br/>- 运算符重载<br/>- 隐藏实现细节"]
    end
    
    subgraph Layer2["第2层：TensorImpl (核心实现层)"]
        TensorImpl["TensorImpl<br/>- 数据管理 (`data_`, `grad_`)<br/>- 计算图管理 (`creator_`, `generation_`)<br/>- 自动求导 (backward())"]
    end
    
    subgraph Layer3["第3层：Mat (抽象接口层)"]
        Mat["Mat (抽象接口)<br/>- 统一矩阵计算接口<br/>- 多态机制<br/>- 后端隔离"]
    end
    
    subgraph Layer4["第4层：后端实现层"]
        OriginMat["OriginMat<br/>(CPU/CUDA)<br/>- Storage 内存管理<br/>- 自定义计算实现"]
        TorchMat["TorchMat<br/>(LibTorch)<br/>- torch::Tensor<br/>- LibTorch 计算"]
    end
    
    UserLayer -->|调用| Layer1
    Layer1 -->|只调用 TensorImpl 方法| Layer2
    Layer2 -->|只调用 Mat 接口方法| Layer3
    Layer3 -->|具体实现| Layer4
    
    Mat -.->|继承| OriginMat
    Mat -.->|继承| TorchMat
    
    style UserLayer fill:#e1f5ff
    style Layer1 fill:#fff4e1
    style Layer2 fill:#ffe1f5
    style Layer3 fill:#e1ffe1
    style Layer4 fill:#f5e1ff
```

**类图（UML 风格）：**

TODO:补充 tensor 的类图

## 2.2 Tensor 层：用户接口层

### 2.2.1 为什么采用值语义？

值语义是指对象的行为像普通值（如 `int`、`double`）一样，拷贝时创建**逻辑上独立**的副本，修改一个对象**不会影响**另一个对象，直观易用，符合直觉。

虽然可以让用户通过智能指针指向 Tensor 实现，但这会导致用户代码中到处都是指针操作，使用起来不够直观。Tensor 是推理框架中大量使用的对象，OriginDL 希望提供与 PyTorch 类似的用户体验，让 Tensor 可以像普通变量一样自然地传递和使用，因此采用了值语义设计。

**用户视角（值语义）**：

```cpp
// OriginDL 采用值语义 + 底层共享的设计策略：
Tensor x({1.0, 2.0, 3.0, 4.0}, Shape({2, 2}), dtype("float32"));
Tensor y = x; // x 和 y 在用户看来是两个独立的对象，实际上底层是浅拷贝，x 和 y 共享同一个 TensorImpl
```

```mermaid
graph LR
    subgraph "用户视角：值语义"
        UserX["Tensor x<br/>独立对象"]
        UserY["Tensor y<br/>独立对象"]
        UserCopy["y = x<br/>逻辑上创建副本"]
        UserX --> UserCopy
        UserCopy --> UserY
    end
    
    subgraph "底层实现：共享机制"
        StackX["Tensor x<br/>(栈上)<br/>`impl_`: shared_ptr"]
        StackY["Tensor y<br/>(栈上)<br/>`impl_`: shared_ptr"]
        SharedImpl["TensorImpl<br/>(堆上)<br/>ref_count=2"]
        StackX -->|"shared_ptr<br/>浅拷贝"| SharedImpl
        StackY -->|"shared_ptr<br/>共享所有权"| SharedImpl
    end
    
    UserCopy -.->|"实际实现"| SharedImpl
    
    style UserX fill:#e1f5ff
    style UserY fill:#e1f5ff
    style UserCopy fill:#fff4e1
    style StackX fill:#ffe1f5
    style StackY fill:#ffe1f5
    style SharedImpl fill:#e1ffe1
```

**Tensor 类的核心设计：**

```cpp
class Tensor {
private:
    std::shared_ptr<TensorImpl> impl_;  // 唯一的成员：智能指针
    
public:
    // 拷贝构造函数：浅拷贝，共享底层 TensorImpl
    Tensor(const Tensor &other) : impl_(other.impl_) {}
    
    // 赋值运算符：浅拷贝，共享底层 TensorImpl
    Tensor &operator=(const Tensor &other) {
        impl_ = other.impl_;
        return *this;
    }
    
    // 移动构造函数：转移所有权，高效
    Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}
    
    // 移动赋值运算符：转移所有权，高效
    Tensor &operator=(Tensor &&other) noexcept {
        impl_ = std::move(other.impl_);
        return *this;
    }
    
    // 运算符重载：返回新 Tensor（不可变设计）
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
};
```

#### 用户接口设计原则

OriginDL 的 Tensor 接口设计遵循以下核心原则：

**值语义与实现隐藏**

Tensor 采用值语义设计，对象可以像普通值一样传递和使用。Tensor 类仅包含一个 `shared_ptr<TensorImpl>` 成员，完全隐藏了 TensorImpl 的实现细节。用户只需要操作 Tensor 接口，无需关心底层实现，这大大降低了使用复杂度。

**不可变接口设计**

所有运算符返回新 Tensor（不可变），例如 `add()`, `transpose()`, `to()` 等方法都返回新实例，不修改原对象，除非显示的使用in-place方法，如以`_`结尾的方法，比如`add_()`。这种设计保证了数据安全，避免了意外修改。其实 Pytorch 也是这么做的。

**链式调用支持**

由于操作返回新的 Tensor，天然支持链式调用，提升代码可读性和表达力：
```cpp
auto y = x.reshape({2, 4}).transpose().to(device);
auto z = x.to(device).to(dtype);
```

**PyTorch 风格 API**

为了降低学习成本并便于从 PyTorch 迁移，OriginDL 提供了与 PyTorch 相似的 API 设计：
- **属性查询**：`shape()`, `dtype()`, `device()`
- **维度信息**：`numel()`, `nbytes()`, `element_size()`
- **梯度管理**：`backward()`, `grad()`, `detach()`, `requires_grad()`

**运算符重载通过算子层**

运算符重载（如 `operator+`, `operator*`）统一通过算子层实现，确保所有运算都能正确建立计算图，支持自动求导。例如 `x + y` 会创建 `Add` 算子，`x * y` 会创建 `Mul` 算子，实现了统一的计算图管理。

#### 运算符重载策略

**运算符重载架构：**

```mermaid
flowchart TD
    %% 用户代码层
    UserCode["用户代码<br/>auto z = x + y;"]
    
    %% 运算符重载层
    OperatorOverload["全局运算符重载<br/>operator+(const Tensor&, const Tensor&)"]
    
    %% 函数式接口层
    FunctionalAPI["函数式接口<br/>functional::add(lhs, rhs)"]
    
    %% 算子创建和调用层
    OperatorCall["算子调用<br/>Add::operator()(xs)"]
    
    %% 算子前向传播层
    Forward["前向传播<br/>Add::forward(xs)"]
    
    %% 计算图管理
    SetupGraph["设置计算图<br/>set_creator()<br/>setup_computation_graph()"]
    
    %% Mat 抽象层
    MatLayer["Mat 抽象层<br/>mat(x0) + mat(x1)"]
    
    %% 返回结果
    Result["返回新 Tensor<br/>包含计算结果和计算图信息"]
    
    %% 流程连接
    UserCode -->|调用| OperatorOverload
    OperatorOverload -->|委托| FunctionalAPI
    FunctionalAPI -->|创建 Add 算子| OperatorCall
    OperatorCall -->|执行| Forward
    Forward -->|调用| MatLayer
    Forward -->|建立| SetupGraph
    MatLayer -->|返回 Mat| Forward
    Forward -->|返回 Tensor| OperatorCall
    SetupGraph -->|设置 `creator_`| Result
    OperatorCall -->|返回| Result
    Result -->|赋值给 z| UserCode
    
    %% 样式
    style UserCode fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style OperatorOverload fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    style FunctionalAPI fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style OperatorCall fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
    style Forward fill:#f5e1ff,stroke:#9900cc,stroke-width:2px
    style SetupGraph fill:#ffffe1,stroke:#999900,stroke-width:2px
    style MatLayer fill:#e1e1ff,stroke:#0000cc,stroke-width:2px
    style Result fill:#ffe1e1,stroke:#cc0000,stroke-width:2px
```

- **值语义接口**：用户代码 `x + y` 直观自然，无需关心底层实现
- **统一通过算子层**：所有运算符重载都创建对应的算子（如 `Add`, `Mul`），不直接操作底层 Mat
- **自动建立计算图**：算子自动设置 `creator_`，建立计算图连接，支持自动求导
- **不可变设计**：所有运算符返回新 Tensor，原对象保持不变
- **类型提升支持**：自动处理不同数据类型的运算，确保类型安全

## 2.3 TensorImpl 层：核心实现

TensorImpl 是 Tensor 的核心实现层，承担数据管理、计算图管理和自动求导三大核心职责：

```mermaid
flowchart TD
    TensorImpl["TensorImpl<br/>核心实现层"]
    
    subgraph Duties["三大核心职责"]
        DataMgmt["数据管理<br/>- 管理 `data_` 和 `grad_`<br/>- 通过 Mat 抽象层访问<br/>- 拷贝时深拷贝保证值语义"]
        GraphMgmt["计算图管理<br/>- 记录 `creator_`<br/>- 记录 `generation_`<br/>- 建立计算图连接"]
        Autograd["自动求导<br/>- backward() 反向传播<br/>- 梯度计算与累积<br/>- 拓扑排序"]
    end
    
    TensorImpl --> Duties
    
    style TensorImpl fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style DataMgmt fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style GraphMgmt fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    style Autograd fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
```

#### 数据成员设计

TensorImpl 的数据成员通过 Mat 抽象层管理数据，通过 Operator 管理计算图：

```mermaid
classDiagram
    class TensorImpl {
        +shared_ptr~Mat~ data_
        +shared_ptr~Mat~ grad_
        +shared_ptr~Operator~ creator_
        +int generation_
    }
    
    class Mat {
        <<abstract>>
        +clone()* unique_ptr~Mat~
        +operator+()* unique_ptr~Mat~
        +shape()* Shape
    }
    
    class Operator {
        <<abstract>>
        +forward()* vector~Tensor~
        +backward()* vector~Tensor~
        +int generation_
    }
    
    class Storage {
        +void* data_
        +size_t size_
    }
    
    class OriginMat {
        +shared_ptr~Storage~ storage_
    }
    
    TensorImpl "1" *-- "1" Mat : data_
    TensorImpl "1" *-- "0..1" Mat : grad_
    TensorImpl "1" --> "0..1" Operator : creator_
    Mat <|-- OriginMat
    OriginMat "1" *-- "1" Storage : storage_
    
    note for TensorImpl "data_: 存储张量实际数据\n通过 Mat 抽象层，支持多后端\n\ngrad_: 存储梯度数据\n初始为 nullptr\nbackward() 时创建\n\ncreator_: 记录创建者 Operator\n用于构建计算图\n\ngeneration_: 记录计算图代数\n用于拓扑排序"
```

#### 核心方法架构

TensorImpl 的核心方法分为反向传播、计算图管理和张量操作三类：

```mermaid
flowchart TD
    subgraph Methods["TensorImpl 核心方法"]
        direction TB
        
        subgraph Backward["反向传播"]
            backward["backward()<br/>反向传播入口<br/>自动构建计算图"]
        end
        
        subgraph GraphMgmt["计算图管理"]
            set_creator["set_creator()<br/>设置创建者 Operator"]
            clear_grad["clear_grad()<br/>清除梯度数据"]
            detach["detach()<br/>断开计算图连接"]
        end
        
        subgraph TensorOps["张量操作"]
            reshape["reshape()<br/>重塑形状"]
            transpose["transpose()<br/>转置操作"]
            to["to()<br/>类型/设备转换"]
        end
        
        subgraph Accessors["访问器方法"]
            shape["shape()<br/>获取形状"]
            dtype["dtype()<br/>获取数据类型"]
            device["device()<br/>获取设备信息"]
        end
    end
    
    style Backward fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style GraphMgmt fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    style TensorOps fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style Accessors fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
```

## 2.4 Mat 抽象层

Mat 是 OriginDL 中的矩阵计算抽象接口层，位于 Tensor 系统四层架构的第三层（Tensor → TensorImpl → Mat → 后端实现），对应 [2.1.1 四层架构概览](#211-四层架构概览)。它的核心作用是：

- **后端解耦**：上层（TensorImpl、Operator）只依赖 Mat 接口，不依赖具体实现
- **多后端支持**：同一接口可对应不同实现，例如 OriginMat（自研 CPU/CUDA 后端）、TorchMat（基于 LibTorch 的后端）
- **统一接口**：所有后端通过同一套 Mat 接口对外提供服务



```shell
# 在整体架构中的位置
Tensor (值语义) → TensorImpl (数据与计算图) → Mat (抽象接口) → OriginMat/TorchMat → Storage (内存)
```

Mat 是 TensorImpl 与具体后端之间的中间层，TensorImpl 通过 `shared_ptr<Mat>` 持有数据，不关心底层是 CPU 还是 GPU、自研还是 LibTorch。

从 `mat.h` 可见，Mat 接口大致包括：

| 类别 | 接口 |
|------|------|
| 拷贝与视图 | `clone()`（深拷贝）、`view()`（共享存储的视图） |
| 形状与连续性 | `reshape()`、`transpose()`、`is_contiguous()`、`contiguous()` |
| 算术运算 | `operator+`、`operator-`、`operator*` 及对应的 `*_inplace` |
| 矩阵运算 | `matmul()`、激活函数等（在子类中实现） |

Mat 与 Storage 的关系

- Mat 可以**拥有** Storage（持有数据），也可以作为**视图**共享 Storage
- 拷贝时通过 `clone()` 创建新的 Mat；若拥有 Storage，则同时创建新的 Storage 并拷贝数据
- 视图（如 reshape、transpose）通过 `view()` 共享同一 Storage，实现零拷贝

## 2.5 后端实现层

对应 [2.1.1 四层架构概览](#211-四层架构概览) 中的**第 4 层**，提供 Mat 接口的具体实现。

（待完善）



## 2.6 内存所有权设计

### 2.6.1 完整内存层次结构

```mermaid
flowchart TB
    Tensor["Tensor 对象<br/>`impl_` (shared_ptr)"]
    
    TensorImpl["TensorImpl 对象<br/>`data_` (shared_ptr&lt;Mat&gt;)<br/>`grad_` (shared_ptr&lt;Mat&gt;)"]
    
    Mat["Mat 对象<br/>shape, dtype, device<br/>`storage_` (shared_ptr)"]
    
    Storage["Storage 对象<br/>`data_` (void*)<br/>`size_`, `device_type_`<br/>`device_index_`"]
    
    subgraph CPU["CPU 内存"]
        CPUData["CPU 数据存储区<br/>(连续内存块)"]
    end
    
    subgraph GPU["GPU 内存"]
        GPUData["GPU 数据存储区<br/>(CUDA 内存)"]
    end
    
    Tensor -.->|共享引用<br/>shared_ptr| TensorImpl
    TensorImpl -.->|`data_` / `grad_`<br/>shared_ptr| Mat
    Mat -.->|共享引用<br/>shared_ptr| Storage
    Storage -.->|`device_type_`=CPU<br/>拥有所有权 void*| CPUData
    Storage -.->|`device_type_`=GPU<br/>拥有所有权 void*| GPUData
    
    style Tensor fill:#cce5ff
    style TensorImpl fill:#ffe1f5
    style Mat fill:#e1ffe1
    style Storage fill:#f5e1ff
    style CPU fill:#e8f5e9
    style GPU fill:#fff3e0
    style CPUData fill:#c8e6c9
    style GPUData fill:#ffe0b2
```

**层次说明：**

1. **Tensor 层**：值语义包装，仅含 `impl_`（`shared_ptr<TensorImpl>`），拷贝时共享 TensorImpl。
2. **TensorImpl 层**：管理 `data_`、`grad_` 和计算图，拷贝时深拷贝 `data_` 和 `grad_`。
3. **Mat 层**：抽象接口隐藏后端，可拥有 Storage 或作为视图，拷贝时调用 clone()。
4. **Storage 层**：数据拥有者，管理原始内存，可被多个 Mat 共享，支持 CPU/GPU。

### 2.6.3 内存优化策略

#### 零拷贝视图（连续张量的 reshape）

```cpp
TensorImpl TensorImpl::reshape(const Shape &shape) const {
    if (data_->is_contiguous()) {
        // 创建视图，零拷贝
        auto view = data_->view(shape);
        return TensorImpl(std::move(view));
    } else {
        // 非连续，需要拷贝
        auto new_data = data_->clone();
        new_data->reshape(shape);
        return TensorImpl(std::move(new_data));
    }
}
```

零拷贝、O(1) 开销，多视图共享 Storage，支持多种形状变换。

#### 原地操作（add_inplace 用于梯度累加）

```cpp
// 梯度累加时使用原地操作
if (!x.impl_->grad_) {
    // 梯度为空，直接共享
    x.impl_->grad_ = gx.impl_->data_;
} else {
    // 梯度不为空，原地累加（避免创建新对象）
    x.impl_->grad_->add_inplace(*gx.impl_->data_);
}
```

不创建新对象、不分配新内存，直接修改现有数据。


```cpp
// 梯度延迟分配
void TensorImpl::backward() {
    if (!grad_) {
        // 只在需要时分配梯度内存
        grad_ = std::shared_ptr<Mat>(Mat_t::zeros(shape(), options()));
    }
    // ...
}第5章：Tensor 构造函数与工厂方法
```

### 2.6.4 类型转换

#### to() 方法（dtype、device 转换）

**接口：**
```cpp
TensorImpl TensorImpl::to(const TensorOptions& options) const;
Tensor Tensor::to(const TensorOptions& options) const;
```

**功能：**
- 转换数据类型（float32 → float64）
- 转换设备（CPU → CUDA）
- 返回新的 Tensor（不修改原对象）

**实现：**
```cpp
TensorImpl TensorImpl::to(const TensorOptions& options) const {
    // 检查是否需要转换
    if (data_->dtype() == options.dtype() && 
        data_->device() == options.device()) {
        return *this;  // 无需转换
    }
    
    // 创建新的 Mat（转换类型或设备）
    auto new_mat = data_->to(options);
    return TensorImpl(std::move(new_mat));
}
```

#### 类型提升规则

**规则：**
- 低精度 → 高精度：自动提升
- 高精度 → 低精度：需要显式转换（可能丢失精度）
- 相同类型：无需转换

**示例：**
```cpp
auto x = Tensor::zeros({2, 3}, dtype("float32"));
auto y = x.to(dtype("float64"));  // float32 → float64
```

#### 设备迁移机制

**CPU ↔ CUDA 迁移：**
```cpp
// CPU → CUDA
auto cpu_tensor = Tensor::zeros({2, 3}, device("cpu"));
auto cuda_tensor = cpu_tensor.to(device("cuda:0"));

// CUDA → CPU
auto back_to_cpu = cuda_tensor.to(device("cpu"));
```

**实现细节：**
- 需要拷贝数据（不同设备）
- 异步传输（CUDA）
- 自动同步（确保数据一致性）

## 2.7 张量打印系统设计

OriginDL的张量打印系统旨在提供清晰、直观的张量数据展示，同时保持与主流深度学习框架的一致性。

### 2.7.1 打印格式设计

**标量张量 (0维)**

```
(1.0)
```

**一维张量 (1维)**

```
[1.0, 2.0, 3.0]
```

**二维张量 (2维)**

```
[[1, 2, 3],
 [4, 5, 6]]
```

**高维张量 (3维及以上，如4维打印如下)**

```
(0,0,.,.) = 
     0       1       2
     3       4       5
(0,1,.,.) = 
     6       7       8
     9      10      11
```

不同的深度学习框架打印 shape(2,3,2,3) 的效果，裸数据均为数组[0, 1, 2, ..., 33, 34, 35]。

```
PyTorch 打印效果：
tensor([[[[ 0.,  1.,  2.],
          [ 3.,  4.,  5.]],

         [[ 6.,  7.,  8.],
          [ 9., 10., 11.]],

         [[12., 13., 14.],
          [15., 16., 17.]]],


        [[[18., 19., 20.],
          [21., 22., 23.]],

         [[24., 25., 26.],
          [27., 28., 29.]],

         [[30., 31., 32.],
          [33., 34., 35.]]]])

LibTorch 打印效果：
(1,1,.,.) = 
  0  1  2
  3  4  5

(2,1,.,.) = 
  18  19  20
  21  22  23

(1,2,.,.) = 
   6   7   8
   9  10  11

(2,2,.,.) = 
  24  25  26
  27  28  29

(1,3,.,.) = 
  12  13  14
  15  16  17

(2,3,.,.) = 
  30  31  32
  33  34  35
[ CPUFloatType{2,3,2,3} ]
```

本人认为 libtorch 的切片风格，再高维张张量中可以更好的看到细节，因此采用了切片的方式，其次 libtorch 的打印内容存在跳跃，与内存布局0~35的连续顺序不合，此时 Pytorch 的打印风格与内存布局相符。 Origin 结合了这两者的风格，打印效果如下所示：

```
(0,0,.,.) = 
     0       1       2
     3       4       5
(0,1,.,.) = 
     6       7       8
     9      10      11
(0,2,.,.) = 
    12      13      14
    15      16      17
(1,0,.,.) = 
    18      19      20
    21      22      23
(1,1,.,.) = 
    24      25      26
    27      28      29
(1,2,.,.) = 
    30      31      32
    33      34      35

 OriginMat(shape=[2, 3, 2, 3], dtype=float32, device=cpu)
```

切片打印的顺序:

```
// LibTorch风格
(0,0,.,.) → (1,0,.,.) → (0,1,.,.) → (1,1,.,.) → (0,2,.,.) → (1,2,.,.)

// OriginDL 风格
(0,0,.,.) → (0,1,.,.) → (0,2,.,.) → (1,0,.,.) → (1,1,.,.) → (1,2,.,.)


内存布局: [0,1,2,3,4,5, 6,7,8,9,10,11, 12,13,14,15,16,17, 18,19,20,21,22,23, ...]
          ↑─────────↑  ↑─────────↑  ↑─────────↑  ↑─────────↑
          (0,0,.,.)   (0,1,.,.)   (0,2,.,.)   (1,0,.,.)


LibTorch风格的访问顺序：
- `(0,0,.,.)` → 内存地址 0-5
- `(1,0,.,.)` → 内存地址 18-23 (跳跃13个元素)
- `(0,1,.,.)` → 内存地址 6-11 (回跳17个元素)
从内存布局上看，OriginDL中打印的相邻切片在内存布局上是相邻的。
```

使用0-based索引，从编程习惯上，更符合符合C++/Python等主流编程语言的索引约定。

```cpp
// OriginDL采用0-based索引，第一个切片索引为 (0,0,.,.)
(0,0,.,.) = 
     0       1       2
     3       4       5

// 而不是LibTorch的1-based索引，第一个切片的索引为 (1,1,.,.)
(1,1,.,.) = 
     0       1       2
     3       4       5
```



# 3. 动态计算图构建

## 3.1 动态计算图架构

动态计算图是 OriginDL 自动求导系统的核心数据结构，用于记录计算过程和依赖关系。计算图由**节点（Tensor）**和**边（Operator）**组成，在前向传播时自动构建。

### 3.1.1 计算图整体架构

```mermaid
flowchart TD
    subgraph NodeLayer["计算图节点层 (Tensor)"]
        direction TB
        Node1["Tensor 节点<br/>- data_: 节点数据<br/>- grad_: 节点梯度<br/>- creator_: 创建者 Operator<br/>- generation_: 拓扑排序代数"]
    end
    
    subgraph EdgeLayer["计算图边层 (Operator)"]
        direction TB
        Edge1["Operator 边<br/>- inputs_: 输入节点<br/>- outputs_: 输出节点 (weak_ptr)<br/>- generation_: 拓扑排序代数<br/>- forward(): 前向传播<br/>- backward(): 反向传播"]
    end
    
    subgraph BuildProcess["前向传播构建过程"]
        direction TB
        Step1["1. 用户代码<br/>auto y = x * 2;"]
        Step2["2. Operator::operator()<br/>- 调用 forward() 计算输出<br/>- 设置 creator_ 建立连接<br/>- setup_computation_graph()"]
        Step3["3. 计算图连接建立<br/>x ──[Mul]──> y"]
    end
    
    subgraph GraphStructures["计算图结构类型"]
        direction LR
        Single["单输入单输出<br/>x → [Op] → y"]
        MultiInput["多输入单输出<br/>x0 ──┐<br/>     ├→ [Op] → y<br/>x1 ──┘"]
        Branch["分支与合并<br/>     ┌→ [Op1] → y1<br/>x ───┤<br/>     └→ [Op2] → y2"]
        Chain["循环结构<br/>x → [Op1] → y1 → [Op2] → y2"]
    end
    
    NodeLayer -.->|通过 creator_ 连接| EdgeLayer
    BuildProcess -->|自动构建| NodeLayer
    BuildProcess -->|自动构建| EdgeLayer
    EdgeLayer -->|支持多种结构| GraphStructures
    
    style NodeLayer fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style EdgeLayer fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    style BuildProcess fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style GraphStructures fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
```

## 3.2 计算图节点与边的连接关系

```mermaid
classDiagram
    class TensorImpl {
        +shared_ptr~Mat~ data_
        +shared_ptr~Mat~ grad_
        +shared_ptr~Operator~ creator_
        +int generation_
    }
    
    class Operator {
        <<abstract>>
        +vector~Tensor~ inputs_
        +vector~weak_ptr~TensorImpl~~ outputs_
        +int generation_
        +forward(inputs)* vector~Tensor~
        +backward(grad_outputs)* vector~Tensor~
        +setup_computation_graph()
    }
    
    class Add {
        +forward(inputs) vector~Tensor~
        +backward(grad_outputs) vector~Tensor~
    }
    
    class Mul {
        +forward(inputs) vector~Tensor~
        +backward(grad_outputs) vector~Tensor~
    }
    
    TensorImpl "1" --> "0..1" Operator : creator_
    Operator "1" --> "*" TensorImpl : inputs_
    Operator "1" --> "*" TensorImpl : outputs_ (weak_ptr)
    Operator <|-- Add : 继承
    Operator <|-- Mul : 继承
    
    note for TensorImpl "节点属性：<br/>- data_: 存储节点数据<br/>- grad_: 存储节点梯度<br/>- creator_: 指向创建该节点的 Operator<br/>- generation_: 用于拓扑排序"
    note for Operator "边属性：<br/>- inputs_: 输入节点列表<br/>- outputs_: 输出节点列表（weak_ptr避免循环引用）<br/>- generation_: 用于拓扑排序<br/>- forward/backward: 前向/反向传播"
```

## 3.3 前向传播时自动构建计算图

在前向传播过程中，计算图会自动构建。当用户执行 `auto y = x * 2;` 时，系统会：

1. **创建算子**：运算符重载创建 `Mul` 算子
2. **执行前向传播**：调用 `Mul::forward()` 计算输出
3. **建立连接**：设置 `y.creator_ = Mul`，建立节点到边的连接
4. **保存信息**：在 `Mul` 中保存 `inputs_` 和 `outputs_`
5. **设置代数**：`y.generation_ = x.generation_ + 1`

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant Op as Operator::operator()
    participant Forward as Operator::forward()
    participant Tensor as TensorImpl
    participant Setup as setup_computation_graph()
    
    User->>Op: auto y = x * 2;
    Op->>Forward: 调用 forward([x])
    Forward->>Forward: 计算 y = x * 2
    Forward-->>Op: 返回 [y]
    Op->>Tensor: y.set_creator(Mul)
    Note over Tensor: y.creator_ = Mul
    Op->>Setup: setup_computation_graph([x], [y])
    Setup->>Setup: inputs_ = [x]
    Setup->>Setup: outputs_ = [y.impl_]
    Setup->>Setup: generation_ = max(x.generation_) + 1
    Setup-->>Op: 完成
    Op-->>User: 返回 y
```

## 3.4 计算图连接机制

#### `creator_` 字段的作用

`creator_` 字段建立了从节点（Tensor）到边（Operator）的连接，使得反向传播时可以从输出节点回溯到创建它的算子。

- **记录创建者**：每个 Tensor 记录创建它的 Operator
- **建立连接**：通过 `creator_` 建立节点到边的连接
- **反向回溯**：反向传播时从 `creator_` 开始回溯整个计算图

#### `generation_` 的作用（拓扑排序）

`generation_` 用于拓扑排序，确保梯度计算的正确顺序：

- **输入节点**：`generation_ = 0`
- **中间节点**：`generation_ = 输入的最大 generation_ + 1`
- **反向传播顺序**：按 `generation_` 从大到小处理（输出端 → 输入端）

```mermaid
flowchart LR
    subgraph Forward["前向传播：generation_ 递增"]
        direction LR
        X["x<br/>generation_=0"]
        Add["Add<br/>generation_=1"]
        Y1["y1<br/>generation_=1"]
        Mul["Mul<br/>generation_=2"]
        Y2["y2<br/>generation_=2"]
        
        X -->|前向| Add
        Add -->|前向| Y1
        Y1 -->|前向| Mul
        Mul -->|前向| Y2
    end
    
    subgraph Backward["反向传播：generation_ 递减"]
        direction RL
        X2["x<br/>generation_=0"]
        Add2["Add<br/>generation_=1"]
        Y1_2["y1<br/>generation_=1"]
        Mul2["Mul<br/>generation_=2"]
        Y2_2["y2<br/>generation_=2"]
        
        Y2_2 -->|反向| Mul2
        Mul2 -->|反向| Y1_2
        Y1_2 -->|反向| Add2
        Add2 -->|反向| X2
    end
    
    style Forward fill:#e1f5ff
    style Backward fill:#ffe1f5
```

## 3.5 计算图结构类型

OriginDL 支持多种计算图结构：

1. **单输入单输出**：最简单的结构，如 `x → [ReLU] → y`
2. **多输入单输出**：多个输入合并，如 `x0, x1 → [Add] → y`
3. **分支与合并**：一个输入产生多个输出，需要梯度累积
4. **循环结构**：链式结构，多个算子串联

### 3.5.1 动态图的优势

- **灵活性**：计算图结构可以动态变化，支持条件分支、循环等控制流
- **易用性**：用户代码直观，不需要显式构建图
- **易调试**：可以打印中间结果，检查计算图结构


# 4. 反向传播实现

## 4.1 反向传播架构

反向传播是自动求导系统的核心，通过遍历计算图从输出端向输入端传播梯度。OriginDL 采用拓扑排序算法确保梯度计算的正确顺序，并通过梯度累积机制处理多路径梯度传播。

### 4.1.1 反向传播整体流程

```mermaid
flowchart TD
    Start["用户调用<br/>y.backward()"]
    
    subgraph Init["初始化阶段"]
        Check["检查 enable_backprop"]
        InitGrad["初始化梯度<br/>grad_ = ones(shape)"]
        InitQueue["初始化队列<br/>funcs, func_set"]
        AddCreator["添加 creator_ 到队列"]
    end
    
    subgraph TopoSort["拓扑排序阶段"]
        Sort["按 generation_ 排序<br/>从大到小"]
        Process["处理 Operator<br/>从队列尾部取出"]
    end
    
    subgraph GradCalc["梯度计算阶段"]
        CollectGrad["收集输出梯度<br/>weak_ptr → shared_ptr"]
        CallBackward["调用 Operator::backward(gys)"]
        GetInputGrad["获取输入梯度 gxs"]
    end
    
    subgraph Accumulate["梯度累积阶段"]
        CheckGrad["检查 grad_ 是否为空"]
        DirectShare["直接共享<br/>grad_ = gx.data_"]
        InplaceAdd["原地累加<br/>grad_.add_inplace(gx)"]
    end
    
    subgraph Recursive["递归处理阶段"]
        AddInputCreator["添加输入 creator_ 到队列"]
        Continue["继续处理下一个 Operator"]
    end
    
    subgraph Cleanup["清理阶段"]
        ClearInputs["清理 inputs_"]
        ClearCreator["清理 creator_"]
        ClearGrad["清理中间 grad_"]
    end
    
    Start --> Check
    Check -->|启用| InitGrad
    Check -->|禁用| End1["直接返回"]
    InitGrad --> InitQueue
    InitQueue --> AddCreator
    AddCreator --> Sort
    Sort --> Process
    Process --> CollectGrad
    CollectGrad --> CallBackward
    CallBackward --> GetInputGrad
    GetInputGrad --> CheckGrad
    CheckGrad -->|为空| DirectShare
    CheckGrad -->|不为空| InplaceAdd
    DirectShare --> AddInputCreator
    InplaceAdd --> AddInputCreator
    AddInputCreator --> Continue
    Continue -->|队列非空| Process
    Continue -->|队列为空| ClearInputs
    ClearInputs --> ClearCreator
    ClearCreator --> ClearGrad
    ClearGrad --> End2["完成"]
    
    style Start fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style Init fill:#fff4e1,stroke:#ff9900,stroke-width:2px
    style TopoSort fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style GradCalc fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
    style Accumulate fill:#f5e1ff,stroke:#9900cc,stroke-width:2px
    style Recursive fill:#ffffe1,stroke:#999900,stroke-width:2px
    style Cleanup fill:#e1e1ff,stroke:#0000cc,stroke-width:2px
```

## 4.2 拓扑排序与梯度计算

```mermaid
flowchart LR
    subgraph Queue["待处理队列 (按 generation_ 排序)"]
        direction TB
        Op3["Operator (gen=3)<br/>输出端"]
        Op2["Operator (gen=2)<br/>中间层"]
        Op1["Operator (gen=1)<br/>中间层"]
        Op0["Operator (gen=0)<br/>输入端"]
    end
    
    subgraph Process["处理流程"]
        direction TB
        Step1["1. 从队列尾部取出<br/>（generation_ 最大）"]
        Step2["2. 收集输出梯度 gys<br/>weak_ptr → shared_ptr"]
        Step3["3. 调用 backward(gys)<br/>计算输入梯度 gxs"]
        Step4["4. 梯度累积<br/>直接共享或原地累加"]
        Step5["5. 添加输入 creator_<br/>递归处理"]
    end
    
    Queue -->|按序处理| Process
    Process -->|递归添加| Queue
    
    style Queue fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style Process fill:#fff4e1,stroke:#ff9900,stroke-width:2px
```

## 4.3 梯度累积机制

梯度累积处理多路径梯度传播的情况。当一个 Tensor 被多个 Operator 使用时，需要将所有路径的梯度累加。

```mermaid
flowchart TD
    subgraph MultiPath["多路径梯度场景"]
        direction LR
        X["x (generation_=0)"]
        Mul["Mul"]
        Y1["y1"]
        Add1["Add"]
        Add2["Add"]
        Y2["y2"]
        Final["y (generation_=2)"]
        
        X -->|路径1| Mul
        Mul -->|路径1| Y1
        Y1 -->|路径1| Add1
        X -->|路径2| Add2
        Add2 -->|路径2| Y2
        Y1 -->|合并| Final
        Y2 -->|合并| Final
    end
    
    subgraph Accumulate["梯度累积过程"]
        direction TB
        Step1["1. 路径1计算<br/>gx1 = backward(y → y1 → x)"]
        Step2["2. 第一次累积<br/>x.grad_ = gx1 (直接共享)"]
        Step3["3. 路径2计算<br/>gx2 = backward(y → y2 → x)"]
        Step4["4. 第二次累积<br/>x.grad_.add_inplace(gx2)"]
        Step5["5. 最终结果<br/>x.grad_ = gx1 + gx2"]
    end
    
    MultiPath -->|反向传播| Accumulate
    
    style MultiPath fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style Accumulate fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
```

## 4.4 梯度累积策略

OriginDL 采用两种梯度累积策略，优化内存使用和性能：

1. **直接共享**：当 `grad_` 为空时，直接共享梯度数据，避免内存分配和拷贝
2. **原地累加**：当 `grad_` 不为空时，使用 `add_inplace()` 原地累加，避免创建新对象

```mermaid
flowchart TD
    Input["输入梯度 gx"]
    Check{"检查 x.grad_<br/>是否为空？"}
    Empty["grad_ 为空"]
    NotEmpty["grad_ 不为空"]
    
    DirectShare["直接共享<br/>x.grad_ = gx.data_<br/>避免拷贝"]
    InplaceAdd["原地累加<br/>x.grad_.add_inplace(gx.data_)<br/>避免分配新内存"]
    
    Result["梯度累积完成"]
    
    Input --> Check
    Check -->|是| Empty
    Check -->|否| NotEmpty
    Empty --> DirectShare
    NotEmpty --> InplaceAdd
    DirectShare --> Result
    InplaceAdd --> Result
    
    style Empty fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
    style NotEmpty fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style DirectShare fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style InplaceAdd fill:#fff4e1,stroke:#ff9900,stroke-width:2px
```

## 4.5 计算图清理机制

反向传播完成后，系统会自动清理计算图，释放内存并断开循环引用：

```mermaid
flowchart TD
    AfterBackward["backward() 完成"]
    
    subgraph Cleanup["清理阶段"]
        direction TB
        ClearInputs["清理 inputs_<br/>Operator.inputs_.clear()"]
        ClearCreator["清理 creator_<br/>TensorImpl.creator_ = nullptr"]
        ClearMidGrad["清理中间 grad_<br/>中间 Tensor 的 grad_ = nullptr"]
        KeepUserGrad["保留用户梯度<br/>输入/输出 Tensor 的 grad_ 保留"]
    end
    
    subgraph Result["清理结果"]
        direction TB
        MemoryFree["内存释放<br/>减少内存占用"]
        BreakCycle["断开循环引用<br/>避免内存泄漏"]
        UserAccess["用户仍可访问<br/>输入/输出梯度"]
    end
    
    AfterBackward --> ClearInputs
    ClearInputs --> ClearCreator
    ClearCreator --> ClearMidGrad
    ClearMidGrad --> KeepUserGrad
    KeepUserGrad --> MemoryFree
    MemoryFree --> BreakCycle
    BreakCycle --> UserAccess
    
    style AfterBackward fill:#e1f5ff,stroke:#0066cc,stroke-width:2px
    style Cleanup fill:#ffe1f5,stroke:#cc0066,stroke-width:2px
    style Result fill:#e1ffe1,stroke:#00cc66,stroke-width:2px
```

### 4.5.1 关键设计要点

1. **拓扑排序**：使用 `generation_` 确保梯度计算顺序正确，从输出端到输入端
2. **去重机制**：使用 `set` 避免重复处理同一个 Operator
3. **weak_ptr 管理**：`outputs_` 使用 `weak_ptr` 避免循环引用，在 `backward()` 时转换为 `shared_ptr`
4. **梯度累积优化**：直接共享和原地累加两种策略，优化内存使用和性能
5. **自动清理**：`backward()` 完成后自动清理计算图，释放内存并断开循环引用

### 4.5.2 反向传播示例

```mermaid
sequenceDiagram
    participant User as 用户
    participant Y as Tensor y
    participant Backward as backward()
    participant Queue as 拓扑排序队列
    participant Op as Operator
    participant X as Tensor x
    
    User->>Y: y.backward()
    Y->>Backward: TensorImpl::backward()
    Backward->>Backward: 初始化 grad_ = ones()
    Backward->>Queue: 添加 creator_ 到队列
    loop 遍历计算图
        Queue->>Op: 取出 Operator (gen最大)
        Op->>Op: 收集输出梯度 gys
        Op->>Op: backward(gys) 计算输入梯度
        Op->>X: 梯度累积到 x.grad_
        Op->>Queue: 添加 x.creator_ 到队列
    end
    Backward->>Backward: 清理计算图
    Backward-->>User: 完成
```



# 5. 算子系统架构

算子系统是 OriginDL 的核心计算组件，负责实现各种数学运算、激活函数、卷积等操作。详细的算子设计理论请参见 [算子系统设计理论](operators_theory.md) 文档。

## 5.1 Operator 基类设计

Operator 是计算图的边，连接输入/输出 Tensor，子类实现 `forward`/`backward`，基类负责调用与计算图维护。

**核心接口：**

| 接口 | 说明 |
|------|------|
| `operator()(inputs)` | 入口：调用 forward → 设置 `creator_` → setup_computation_graph |
| `forward(inputs)` | 纯虚，前向计算 |
| `backward(grad_outputs)` | 纯虚，反向梯度 |
| `forward_inplace(input0, input1)` | 可选原地前向，默认抛异常 |

**设计要点：**

- `enable_shared_from_this`：用于设置输出的 `creator_`
- `outputs_` 用 `weak_ptr<TensorImpl>` 避免 Operator ↔ TensorImpl 循环引用
- `kNullTensor_`：区分一元/二元原地操作（`input1 == kNullTensor_` 表示一元）

**调用流程：**

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant Op as Operator::operator()
    participant Fwd as forward()
    participant Impl as TensorImpl

    User->>Op: operator()(xs)
    Op->>Fwd: forward(xs)
    Fwd->>Fwd: Mat 计算
    Fwd-->>Op: outputs
    Op->>Impl: set_creator(Op)
    Op->>Op: setup_computation_graph(xs, outputs)
    Op-->>User: outputs
```

## 5.2 类型提升

算子层统一通过 `TypePromotion` 处理不同 dtype 的运算，以 Add 为例。

**规则：** 优先级 float64 > float32 > int64 > int32 > int8 > uint8，取两者中更高精度。

**Add forward：**
```cpp
auto [x0, x1] = TypePromotion::promote_tensors_maybe_owned(xs[0], xs[1]);
auto result = mat(x0) + mat(x1);
```
**Add backward：** 梯度与 forward 输出类型一致，无需再提升，直接传递 `gys[0]`。

**forward_inplace：** 需原地修改 `input0`，先 `input0.to(promoted_type)`，再对 `input1` 用 `to_type_maybe_owned` 转换后执行 `add_inplace`。

**MaybeOwned 设计：** 参考 PyTorch c10::MaybeOwned，用于类型提升的零开销优化。两种模式：`borrowed` 只存指针不增加引用计数；`owned` 持有 `unique_ptr` 拥有新对象。类型匹配时借用，不匹配时仅对需转换的 Tensor 创建新对象。支持隐式转换为 `T&`，便于 `mat(x0)` 等用法。

```mermaid
flowchart TB
    Check{"a.dtype() == b.dtype()?"}
    Same["borrowed(a), borrowed(b)<br/>零开销"]
    Diff["promoted = promote_types(a, b)"]
    ToBoth["对 a、b 分别调用 to_type_maybe_owned"]
    Match{"tensor.dtype() == target?"}
    Borrow["borrowed: 借用引用"]
    Own["owned: 创建新对象"]

    Check -->|是| Same
    Check -->|否| Diff
    Diff --> ToBoth
    ToBoth --> Match
    Match -->|是| Borrow
    Match -->|否| Own
```

```mermaid
flowchart LR
    subgraph Borrowed["borrowed 模式"]
        direction TB
        B_ptr["ptr_ 指向原对象"]
        B_owned["is_owned_ = false"]
        B_note["不增加引用计数"]
    end

    subgraph Owned["owned 模式"]
        direction TB
        O_ptr["owned_ptr_ 持有 unique_ptr"]
        O_get["ptr_ = owned_ptr_.get()"]
        O_owned["is_owned_ = true"]
    end
```

# 6. 神经网络模块架构

## 6.1 Module 基类设计

```mermaid
classDiagram
    class Module {
        <<abstract>>
        #parameters_ map~string,Parameter*~
        #modules_ map~string,unique_ptr~Module~~
        #training_ bool
        +forward(input)* Tensor
        +operator()(input) Tensor
        +parameters() vector~Parameter*~
        +register_parameter(name, param) void
        +register_module(name, module) void
        +state_dict() StateDict
        +load_state_dict(state_dict) void
        +train(mode) void
        +eval() void
        +to(device) void
        +zero_grad() void
    }
```

Module 提供参数/子模块注册、递归收集、train/eval 模式、设备迁移、梯度清零。`forward` 为纯虚，`operator()` 委托调用。

## 6.2 Layer 层设计

```mermaid
classDiagram
    class Tensor {
        +shape() Shape
        +dtype() DataType
    }
    class Parameter {
        <<extends Tensor>>
    }
    class Module {
        <<abstract>>
    }
    class Layer {
        <<extends Module>>
    }
    class Linear {
        -weight_ Parameter
        -bias_ Parameter
        +forward(input) Tensor
    }
    class ReLU {
        +forward(input) Tensor
    }

    Tensor <|-- Parameter
    Module <|-- Layer
    Layer <|-- Linear
    Layer <|-- ReLU
    Linear --> Parameter : 持有
```

Layer 继承 Module，作为带参数层的基类。Parameter 继承 Tensor 标识可训练参数。有参数层（Linear、Conv2d）持有 `weight_`/`bias_` 并 `register_parameter`；无参数层（ReLU、Flatten）仅实现 `forward`。

## 6.3 Sequential 容器设计

```mermaid
flowchart LR
    subgraph Sequential
        M0["modules_[0]"]
        M1["modules_[1]"]
        M2["modules_[2]"]
        Mn["..."]
    end

    Input["input"] --> M0 --> M1 --> M2 --> Mn --> Output["output"]
```

```mermaid
classDiagram
    class Module {
        <<abstract>>
    }
    class Sequential {
        -modules_ vector~unique_ptr~Module~~
        +add(module) void
        +forward(input) Tensor
        +operator[](index) Module
        +parameters() vector~Parameter*~
        +to(device) void
    }
    Module <|-- Sequential
    Sequential *-- Module : 顺序持有
```

Sequential 用 `vector<unique_ptr<Module>>` 顺序存储子模块，`forward` 依次调用，`parameters`/`to` 递归聚合。

## 6.4 常用层实现

```mermaid
flowchart TB
    subgraph 有参数层
        Linear["Linear<br/>y = xW + b"]
        Conv2d["Conv2d<br/>y = conv2d(x,W) + b"]
        BN["BatchNorm1d/2d"]
    end

    subgraph 无参数层
        ReLU["ReLU"]
        MaxPool2d["MaxPool2d"]
        Flatten["Flatten"]
        Dropout["Dropout"]
    end
```

| 层 | 参数 | 对应算子 |
|----|------|----------|
| Linear | weight, bias | matmul, add |
| Conv2d | weight, bias | conv2d |
| ReLU / Flatten / Dropout | 无 | relu / flatten / dropout |
| MaxPool2d | 无 | max_pool2d |
| BatchNorm1d/2d | gamma, beta, running_mean/var | batch_norm |

# 7. 优化器架构

## 7.1 Optimizer 基类设计

```mermaid
classDiagram
    class Optimizer {
        <<abstract>>
        #target_ Module*
        #hooks_ vector~function~
        #parameters_ vector~Parameter*~
        +Optimizer(target) 
        +step() void
        +zero_grad() void
        +register_hook(hook) void
        +parameters() vector~Parameter*~
        +state_dict()* map
        +load_state_dict()* void
        #step_one(param)* void
    }
```

```mermaid
flowchart TB
    Step["step()"]
    Filter["过滤有梯度的参数"]
    Hooks["执行 hooks_"]
    StepOne["对每个 param 调用 step_one()"]

    Step --> Filter --> Hooks --> StepOne
```

Optimizer 持有 Module 引用，构造时 `collect_parameters` 收集参数。`step` 流程：过滤有梯度参数 → 执行 hooks → 逐参数 `step_one`。`zero_grad` 委托给 target。

```cpp
// 典型训练循环（假设 using namespace origin）
auto model = Sequential();
model.add(std::make_unique<nn::Linear>(784, 10, true));
auto optimizer = Adam(model, 0.01f);

for (int i = 0; i < epochs; ++i) {
    optimizer.zero_grad();      // 委托 target_->zero_grad()
    auto y = model(x);
    auto loss = functional::softmax_cross_entropy(y, target);
    loss.backward();
    optimizer.step();           // 过滤有梯度参数 → hooks → step_one
}
```

## 7.2 SGD 优化器实现

```mermaid
classDiagram
    class Optimizer {
        <<abstract>>
    }
    class SGD {
        -lr_ float
        -momentum_ float
        -weight_decay_ float
        -nesterov_ bool
        -momentum_buffers_ map
        #step_one(param) void
    }
    Optimizer <|-- SGD
```

| 参数 | 默认 | 说明 |
|------|------|------|
| lr | - | 学习率 |
| momentum | 0 | 动量 |
| weight_decay | 0 | L2 正则 |
| nesterov | false | Nesterov 动量 |

SGD 公式：`param = param - lr * (grad + weight_decay * param)`，momentum 时维护 `momentum_buffers_`。

## 7.3 Adam 优化器实现

```mermaid
classDiagram
    class Optimizer {
        <<abstract>>
    }
    class Adam {
        -lr_ float
        -beta1_ float
        -beta2_ float
        -eps_ float
        -m_buffers_ map
        -v_buffers_ map
        -step_counts_ map
        #step_one(param) void
    }
    Optimizer <|-- Adam
```

```mermaid
flowchart LR
    subgraph Adam公式
        M["m = β1·m + (1-β1)·g"]
        V["v = β2·v + (1-β2)·g²"]
        Mh["m̂ = m/(1-β1^t)"]
        Vh["v̂ = v/(1-β2^t)"]
        P["param -= lr·m̂/(√v̂+ε)"]
    end
    M --> Mh
    V --> Vh
    Mh --> P
    Vh --> P
```

Adam 维护一阶矩 `m_buffers_`、二阶矩 `v_buffers_`、步数 `step_counts_`，支持 state_dict 保存/加载。

## 7.4 优化器钩子机制

```mermaid
flowchart LR
    Step["step()"]
    Filter["过滤有梯度参数"]
    Hooks["遍历 hooks_"]
    Hook["hook(params)"]
    StepOne["step_one()"]

    Step --> Filter --> Hooks --> Hook --> StepOne
```

```mermaid
classDiagram
    class WeightDecay {
        -rate_ float
        +hook() function
    }
    note for WeightDecay "grad += rate * param\n等价于 L2 正则"
```

Hook 在 `step_one` 之前执行，签名为 `void(vector<Parameter*>&)`。WeightDecay 对梯度累加 `rate * param`，实现 L2 正则。

# 8. 数据处理架构

## 8.1 Dataset 接口设计

## 8.2 DataLoader 实现

## 8.3 数据预处理机制

# 9. IO 模块架构

## 9.1 Checkpoint 机制

## 9.2 Model IO 设计

## 9.3 模型保存与加载

# 10. PNNX 推理架构

## 10.1 PNNX 图结构

## 10.2 模型推理流程

## 10.3 算子映射机制

# 11. 应用示例

本章介绍 OriginDL 框架在实际应用中的使用示例，包括线性回归、MNIST 训练、ResNet 分类和 YOLOv5 推理等典型场景。

## 11.1 线性回归示例

（待完善）

## 11.2 MNIST 训练示例

（待完善）

### 11.2.1 MLP 模型训练

（待完善）

### 11.2.2 CNN 模型训练

（待完善）

## 11.3 ResNet 分类推理

（待完善）

## 11.4 YOLOv5 目标检测推理

（待完善）
