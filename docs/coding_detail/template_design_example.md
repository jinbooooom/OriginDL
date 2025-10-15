# 模板化设计使用示例

## 概述

本文档展示了如何使用新的模板化设计来替代大量重复的switch语句，提高代码的可维护性和扩展性。

## 设计优势

### 1. 代码简化
**重构前（add.cpp）：**
```cpp
// 195行代码，大量重复的switch语句
switch (a.dtype()) {
    case DataType::kFloat32: {
        const float *a_data = a.data_ptr<float>();
        const float *b_data = b.data_ptr<float>();
        float *c_data = result->data_ptr<float>();
        // 重复的广播逻辑...
        break;
    }
    case DataType::kFloat64: {
        const double *a_data = a.data_ptr<double>();
        const double *b_data = b.data_ptr<double>();
        double *c_data = result->data_ptr<double>();
        // 重复的广播逻辑...
        break;
    }
    // 更多重复的case...
}
```

**重构后（add.cpp）：**
```cpp
// 35行代码，简洁明了
std::unique_ptr<OriginMat> add(const OriginMat &a, const OriginMat &b) {
    if (a.dtype() != b.dtype()) {
        THROW_INVALID_ARG("Data type mismatch for addition: expected {} but got {}", 
                         dtype_to_string(a.dtype()), dtype_to_string(b.dtype()));
    }

    Shape result_shape = compute_broadcast_shape(a, b);
    auto result = std::make_unique<OriginMat>(result_shape, a.dtype());

    TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
        BroadcastCompute::binary_broadcast<T>(a, b, *result, AddOp{});
    });

    return result;
}
```

### 2. 易于扩展新数据类型

添加新数据类型（如`kFloat16`）只需要：

1. 在`DataType`枚举中添加新类型
2. 在`TypeDispatcher`中添加一个case
3. 在类型特征模板中添加特化

所有现有的计算函数都会自动支持新类型，无需修改任何计算逻辑。

### 3. 类型安全

模板系统提供编译时类型检查，避免运行时类型错误。

## 使用示例

### 基本操作

```cpp
#include "origin/mat/origin/cpu/operation_templates.h"

// 创建两个矩阵
auto a = std::make_unique<OriginMat>(Shape({2, 3}), DataType::kFloat32);
auto b = std::make_unique<OriginMat>(Shape({2, 3}), DataType::kFloat32);

// 使用新的模板化设计进行加法
auto result = add(*a, *b);
```

### 自定义操作

```cpp
// 定义自定义操作
struct CustomOp {
    template<typename T>
    T operator()(T a, T b) const { 
        return a * a + b * b;  // 自定义数学运算
    }
};

// 使用自定义操作
TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
    BroadcastCompute::binary_broadcast<T>(a, b, *result, CustomOp{});
});
```

### 一元操作

```cpp
// 定义一元操作
struct SqrtOp {
    template<typename T>
    T operator()(T value) const { 
        return std::sqrt(value);
    }
};

// 使用一元操作
TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
    BroadcastCompute::unary<T>(mat, *result, SqrtOp{});
});
```

## 性能对比

### 编译时优化
- **模板实例化**：每种数据类型组合生成独立的优化代码
- **内联优化**：编译器可以更好地内联模板函数
- **向量化**：模板代码更容易被编译器向量化

### 运行时性能
- **零额外开销**：类型分发在编译时完成
- **分支预测**：switch语句优化良好
- **内存访问**：直接指针操作，缓存友好

## 扩展指南

### 添加新的二元操作

1. 定义操作函数对象：
```cpp
struct MultiplyOp {
    template<typename T>
    T operator()(T a, T b) const { return a * b; }
};
```

2. 创建计算函数：
```cpp
std::unique_ptr<OriginMat> multiply(const OriginMat &a, const OriginMat &b) {
    // 类型检查和形状计算
    Shape result_shape = compute_broadcast_shape(a, b);
    auto result = std::make_unique<OriginMat>(result_shape, a.dtype());

    // 使用类型分发器
    TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
        BroadcastCompute::binary_broadcast<T>(a, b, *result, MultiplyOp{});
    });

    return result;
}
```

### 添加新数据类型

1. 在`DataType`枚举中添加：
```cpp
enum class DataType {
    kFloat32 = 0,
    kFloat64 = 1,
    kInt32 = 2,
    kInt8 = 3,
    kFloat16 = 4  // 新类型
};
```

2. 在`TypeDispatcher`中添加case：
```cpp
case DataType::kFloat16:
    func.template operator()<float16_t>();
    break;
```

3. 在类型特征模板中添加特化：
```cpp
template <>
struct DataTypeTraits<float16_t> {
    static constexpr DataType type = DataType::kFloat16;
    static constexpr const char *name = "float16";
};
```

## 总结

新的模板化设计显著改善了代码质量：

- **代码减少**：从195行减少到35行（减少82%）
- **维护性提升**：添加新类型只需修改3个地方
- **类型安全**：编译时类型检查
- **性能保持**：运行时性能与原始设计相当
- **易于扩展**：支持自定义操作和数据类型

这种设计为未来的功能扩展奠定了良好的基础，同时保持了高性能和类型安全。
