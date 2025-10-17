# OriginMat 模板化设计与CUDA支持

## 概述

本文档展示了OriginMat的模板化设计如何替代大量重复的switch语句，提高代码的可维护性和扩展性。同时介绍了OriginMat的CUDA支持实现以及与主库的编译分离设计。

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

添加新数据类型只需要：

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
    kInt16 = 4  // 新类型
};
```

2. 在`TypeDispatcher`中添加case：
```cpp
case DataType::kInt16:
    func.template operator()<int16_t>();
    break;
```

3. 在类型特征模板中添加特化：
```cpp
template <>
struct DataTypeTraits<int16_t> {
    static constexpr DataType type = DataType::kInt16;
    static constexpr const char *name = "int16";
};
```

## CUDA支持实现

### 1. 编译分离设计

OriginMat采用了编译分离的设计，将CUDA支持作为独立的子项目：

```
src/mat/origin/cuda/
├── CMakeLists.txt          # CUDA子项目的构建配置
├── add.cu                  # CUDA加法算子实现
├── cuda_utils.cpp          # CUDA工具函数
├── cuda_ops.h              # CUDA算子接口声明
└── cuda_utils.h            # CUDA工具函数声明
```

**主要特点：**
- **独立编译**：CUDA代码编译为独立的`originmat_cuda.so`库
- **可选依赖**：通过`ENABLE_CUDA`选项控制是否编译CUDA支持
- **自动架构检测**：使用`nvidia-smi`自动检测GPU计算能力
- **版本兼容**：支持不同CUDA版本的C++标准

### 2. CUDA算子实现

**模板化CUDA内核：**
```cpp
template <typename T>
__global__ void add_kernel(const T *a, const T *b, T *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
```

**运行时类型分发：**
```cpp
void dispatch_add(DataType dtype, const void *a, const void *b, void *c, size_t n)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            launch_add_kernel<float>(...);
            break;
        case DataType::kFloat64:
            launch_add_kernel<double>(...);
            break;
        // 支持更多数据类型...
    }
}
```

### 3. 设备管理

**设备类型支持：**
```cpp
// 创建CUDA张量
auto a = std::make_unique<OriginMat>(shape, dtype, Device(DeviceType::kCUDA, 0));

// 设备转换
auto cpu_tensor = cuda_tensor.to(Device(DeviceType::kCPU, 0));
```

**自动设备选择：**
```cpp
// OriginMat::operator+ 实现中的设备选择逻辑
std::unique_ptr<Mat> OriginMat::operator+(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::add(*this, other_mat);  // CPU实现
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::add(*this, other_mat);  // CUDA实现
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for addition");
    }
}
```

**内存管理：**
- **CUDA分配器**：实现`CUDAAllocator`类管理GPU内存
- **设备间传输**：支持CPU↔CUDA内存拷贝
- **自动同步**：操作完成后自动同步设备状态

### 4. 构建系统

**主CMakeLists.txt配置：**
```cmake
# CUDA支持选项
option(ENABLE_CUDA "Enable CUDA support for OriginMat" OFF)

# 如果启用CUDA，添加CUDA子项目
if(ENABLE_CUDA)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/mat/origin/cuda)
    add_definitions(-DWITH_CUDA)
endif()
```

**CUDA子项目配置：**
```cmake
# 自动检测CUDA架构
function(detect_cuda_architecture)
    execute_process(
        COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
        OUTPUT_VARIABLE GPU_COMPUTE_CAP
    )
    # 解析计算能力并设置架构
endfunction()

# 创建独立的CUDA库
add_library(originmat_cuda SHARED ${CUDA_SRCS} ${CUDA_CPP_SRCS})
```

### 5. 使用示例

**编译支持CUDA的版本：**
```bash
# 编译OriginMat后端并启用CUDA支持
./build.sh ORIGIN --cuda

# 指定CUDA编译器路径
./build.sh ORIGIN --cuda --nvcc /usr/local/cuda-12.8/bin/nvcc
```

**CUDA张量操作：**
```cpp
#include "origin.h"

// 检查CUDA可用性
if (origin::cuda::is_cuda_available()) {
    // 创建CUDA张量
    origin::Tensor a({1.0f, 2.0f, 3.0f}, origin::Shape{3}, 
                     origin::dtype(origin::Float32).device(origin::kCUDA));
    origin::Tensor b({4.0f, 5.0f, 6.0f}, origin::Shape{3}, 
                     origin::dtype(origin::Float32).device(origin::kCUDA));
    
    // CUDA加法运算
    auto c = a + b;  // 自动选择CUDA实现
    c.print("CUDA Result");
}
```

## 编译分离的优势

### 1. 构建灵活性
- **可选编译**：用户可以选择是否编译CUDA支持
- **依赖隔离**：CUDA依赖不会影响CPU版本的构建
- **版本兼容**：支持不同CUDA版本和GPU架构

### 2. 部署便利性
- **独立库**：`originmat_cuda.so`可以独立分发
- **按需加载**：运行时根据设备可用性加载相应库
- **减少体积**：CPU版本不包含CUDA代码

### 3. 开发效率
- **并行开发**：CPU和CUDA代码可以并行开发
- **独立测试**：可以独立测试CPU和CUDA功能
- **渐进迁移**：可以逐步添加CUDA算子支持

## 总结

OriginMat的模板化设计和CUDA支持实现了以下目标：

### 代码质量提升
- **代码减少**：从195行减少到35行（减少82%）
- **维护性提升**：添加新类型只需修改3个地方
- **类型安全**：编译时类型检查
- **性能保持**：运行时性能与原始设计相当

### CUDA支持特性
- **编译分离**：独立的CUDA子项目，可选编译
- **自动架构检测**：智能检测GPU计算能力
- **设备管理**：完整的CPU/CUDA设备支持
- **内存管理**：高效的设备间数据传输

### 扩展性设计
- **易于扩展**：支持自定义操作和数据类型
- **模块化**：CPU和CUDA实现完全分离
- **向后兼容**：保持API一致性

这种设计为未来的功能扩展奠定了良好的基础，同时保持了高性能、类型安全和部署灵活性。
