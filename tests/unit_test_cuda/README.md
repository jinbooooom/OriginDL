# CUDA单元测试

本目录包含OriginDL项目的CUDA单元测试，与CPU测试完全独立。

## 目录结构

```
tests/unit_test_cuda/
├── CMakeLists.txt                    # CUDA测试CMake配置
├── test_cuda_utils.cpp              # CUDA工具函数测试
├── test_cuda_tensor.cpp             # CUDA张量基础功能测试
├── test_cuda_memory.cpp             # CUDA内存管理测试
├── operators/                       # CUDA算子测试
│   ├── test_cuda_add.cpp            # CUDA加法算子测试
│   ├── test_cuda_subtract.cpp       # CUDA减法算子测试
│   ├── test_cuda_multiply.cpp       # CUDA乘法算子测试
│   ├── test_cuda_divide.cpp         # CUDA除法算子测试
│   ├── test_cuda_exp.cpp            # CUDA指数算子测试
│   ├── test_cuda_log.cpp            # CUDA对数算子测试
│   ├── test_cuda_sqrt.cpp           # CUDA平方根算子测试
│   ├── test_cuda_square.cpp         # CUDA平方算子测试
│   ├── test_cuda_negate.cpp         # CUDA取负算子测试
│   ├── test_cuda_add_scalar.cpp     # CUDA标量加法测试
│   └── test_cuda_multiply_scalar.cpp # CUDA标量乘法测试
├── performance/                     # CUDA性能测试
│   ├── test_cuda_performance.cpp    # CUDA性能基准测试
│   ├── benchmark_cuda_operators.cpp # CUDA算子性能对比
│   └── test_cuda_memory_performance.cpp # CUDA内存性能测试
├── integration/                     # CUDA集成测试
│   ├── test_cuda_autograd.cpp       # CUDA自动微分测试
│   ├── test_cuda_computation_graph.cpp # CUDA计算图测试
│   └── test_cuda_mixed_precision.cpp # CUDA混合精度测试
└── utils/                           # CUDA测试工具
    ├── cuda_test_utils.h            # CUDA测试工具头文件
    ├── cuda_test_utils.cpp          # CUDA测试工具实现
    └── test_data_generator.h        # 测试数据生成器
```

## 测试分类

### 1. 基础功能测试
- **CUDA工具函数测试** (`test_cuda_utils.cpp`)
  - CUDA设备检测
  - 线程块和网格大小计算
  - CUDA错误处理
  - 内存管理
  - 流管理

- **CUDA张量测试** (`test_cuda_tensor.cpp`)
  - CUDA张量创建
  - 数据类型支持
  - 形状操作
  - 设备管理

- **CUDA内存测试** (`test_cuda_memory.cpp`)
  - 内存分配和释放
  - 内存拷贝
  - 内存泄漏检测

### 2. 算子测试 (`operators/`)
每个CUDA算子都有对应的测试文件，包含：

- **基础功能测试**
  - 基本运算正确性
  - 不同数据类型支持
  - 不同张量形状支持

- **数值精度测试**
  - 浮点精度验证
  - 数值稳定性测试
  - 特殊值处理

- **边界情况测试**
  - 单元素张量
  - 大张量
  - 零值、负值、极值

- **错误处理测试**
  - 无效输入处理
  - 设备不匹配
  - 数据类型不匹配

### 3. 性能测试 (`performance/`)
- **性能基准测试**
  - 不同数据大小的性能表现
  - 与CPU版本的性能对比
  - 内存使用效率测试

- **性能回归测试**
  - 确保性能不退化
  - 性能阈值验证

### 4. 集成测试 (`integration/`)
- **自动微分测试**
  - CUDA张量的梯度计算
  - 计算图构建
  - 反向传播

- **混合精度测试**
  - 不同精度间的转换
  - 精度损失验证

## 测试工具

### 测试基类
- `CudaTestBase`: 基础CUDA测试类
- `CudaPerformanceTest`: 性能测试专用基类

### 工具函数
- 张量比较函数
- 随机数据生成
- 性能测量工具
- CUDA错误检查

### 数据生成器
- 均匀分布数据
- 特殊值数据
- 边界值数据
- 预设测试用例

## 编译和运行

### 编译CUDA测试
```bash
# 确保启用CUDA支持
./build.sh ORIGIN --cuda

# 编译CUDA测试
cd build
make test_cuda_add  # 编译特定测试
make test_cuda_all  # 编译所有CUDA测试
```

### 运行CUDA测试
```bash
# 运行单个测试
./bin/unit_test_cuda/test_cuda_add

# 运行所有CUDA测试
cd build
make run_cuda_unit_tests

# 使用CTest运行
ctest -R cuda
```

### 性能测试
```bash
# 运行性能测试
./bin/unit_test_cuda/test_cuda_performance

# 运行基准测试
./bin/unit_test_cuda/benchmark_cuda_operators
```

## 测试编写指南

### 1. 使用测试基类
```cpp
#include "../utils/cuda_test_utils.h"

class MyCudaTest : public CudaTestBase {
protected:
    void SetUp() override {
        CudaTestBase::SetUp();
        // 自定义设置
    }
};
```

### 2. 创建CUDA张量
```cpp
// 使用工具函数创建CUDA张量
auto tensor = createCudaTensor({1.0f, 2.0f, 3.0f}, {3});
```

### 3. 验证结果
```cpp
// 验证张量属性
verifyTensorShape(*result, {3});
verifyTensorDtype(*result, DataType::kFloat32);
verifyTensorDevice(*result, DeviceType::kCUDA);

// 验证数值结果
auto result_data = result->to_vector<float>();
EXPECT_NEAR(result_data[0], expected_value, kFloatTolerance);
```

### 4. 性能测试
```cpp
class MyPerformanceTest : public CudaPerformanceTest {
    TEST_F(MyPerformanceTest, PerformanceTest) {
        auto operation = [&]() {
            // 执行CUDA操作
        };
        
        float avg_time = measureAverageTime(operation, 10);
        verifyPerformance(avg_time, max_time_ms, "Operation name");
    }
};
```

## 注意事项

1. **环境要求**: 需要CUDA环境和GPU支持
2. **条件编译**: 测试仅在`WITH_CUDA`启用时编译
3. **设备检查**: 测试会自动检查CUDA可用性
4. **内存管理**: 注意CUDA内存的分配和释放
5. **错误处理**: 使用提供的错误检查工具
6. **性能考虑**: 性能测试需要多次运行取平均值

## 扩展测试

添加新的CUDA测试时：

1. 在相应目录创建测试文件
2. 在`CMakeLists.txt`中添加编译配置
3. 使用提供的测试工具和基类
4. 遵循现有的测试模式和命名规范
5. 添加适当的文档和注释
