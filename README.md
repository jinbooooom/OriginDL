# OriginDL: 从零开始构建的分布式深度学习框架

OriginDL 是一个从零开始构建的分布式深度学习框架，采用 C++ 实现，支持自动求导和多种计算后端。项目使用 ArrayFire 作为底层计算引擎，提供了类似 PyTorch 的 API 接口。

## ✨ 特性

- 🚀 **自动求导** - 支持动态计算图和反向传播
- 🔧 **多后端支持** - 基于 ArrayFire，支持 CPU、CUDA、OpenCL
- 📦 **简洁 API** - 类似 PyTorch 的直观接口
- 🎯 **教育友好** - 从零构建，便于理解深度学习框架原理
- 🧪 **完整测试** - 包含单元测试和与 PyTorch 的对比验证

## 🚀 快速开始

### 安装 ArrayFire

```bash
# 下载并安装 ArrayFire
wget https://arrayfire.s3.amazonaws.com/3.9.0/ArrayFire-v3.9.0_Linux_x86_64.sh
sudo sh ArrayFire-v3.9.0_Linux_x86_64.sh --skip-license --prefix=/opt/arrayfire

cd 3rd
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

### 编译项目

```bash
# 设置环境变量
source setEnv.sh

# 编译项目
bash ./build.sh
```

编译完成后，会在以下位置生成文件：
- `build/libs/origindl.so` - 主库文件
- `build/bin/` - 测试程序

## 📖 基本使用

### 创建张量

```cpp
#include "originDL.h"
using namespace dl;

// 从数据创建张量
auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

// 创建零张量
auto zeros = Tensor::zeros(Shape{3, 3});

// 创建随机张量
auto rand_tensor = Tensor::randn(Shape{2, 2});

// 标量张量
auto scalar = Tensor(5.0, Shape{1});
```

### 基本运算

```cpp
auto x = Tensor({1.0, 2.0}, Shape{2});
auto y = Tensor({3.0, 4.0}, Shape{2});

// 基本算术运算
auto z1 = x + y;  // 加法
auto z2 = x - y;  // 减法
auto z3 = x * y;  // 元素级乘法
auto z4 = x / y;  // 除法

// 数学函数
auto z5 = exp(x);     // 指数函数
auto z6 = square(x);  // 平方
auto z7 = x ^ 2;      // 幂运算
```

### 自动求导

```cpp
// 创建需要梯度的张量
auto x = Tensor({2.0, 3.0}, Shape{2});
auto y = Tensor({1.0, 2.0}, Shape{2});

// 前向传播
auto z = x * y + exp(x);

// 反向传播
z.backward();

// 获取梯度
x.grad().print("dx: ");  // 对x的梯度
y.grad().print("dy: ");  // 对y的梯度
```

## 🔧 支持的算子

### 算术算子
- `+` - 加法
- `-` - 减法  
- `*` - 元素级乘法
- `/` - 除法
- `^` - 幂运算

### 数学函数
- `exp()` - 指数函数
- `square()` - 平方
- `sum()` - 求和
- `neg()` - 取负

### 形状操作
- `reshape()` - 重塑形状
- `transpose()` - 转置

## 📝 示例代码

### 线性回归示例

```cpp
#include "originDL.h"
using namespace dl;

int main() {
    // 初始化ArrayFire
    af::setBackend(AF_BACKEND_CPU);
    
    // 创建训练数据
    auto x = Tensor::randn(Shape{100, 1});
    auto y = 2.0 * x + 1.0 + Tensor::randn(Shape{100, 1}) * 0.1;
    
    // 模型参数
    auto w = Tensor::randn(Shape{1, 1});
    auto b = Tensor::zeros(Shape{1, 1});
    
    // 训练循环
    for (int epoch = 0; epoch < 100; ++epoch) {
        // 前向传播
        auto pred = x * w + b;
        auto loss = sum(square(pred - y));
        
        // 反向传播
        loss.backward();
        
        // 更新参数
        w = w - 0.01 * w.grad();
        b = b - 0.01 * b.grad();
        
        // 清除梯度
        w.clear_grad();
        b.clear_grad();
        
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch << ", Loss: " << loss.item() << std::endl;
        }
    }
    
    return 0;
}
```

## 🧪 运行测试

项目包含完整的单元测试，可以验证功能正确性：

```bash
# 编译
$ bash build.sh
# 编译成功后，会在 ./build/bin/ 目录下生成 demo 与单元测试程序。

# 运行线性回归 demo
$ ./build/bin/dl_linearRegression 
ArrayFire v3.9.0 (CPU, 64-bit Linux, build b59a1ae53)
[0] Intel: Intel(R) Core(TM) i7-14700
JinboBook 2025-09-29 21:23:43.035 W 49297 49297 [main.cpp:SetBackend:71] Active Backend: CPU
JinboBook 2025-09-29 21:23:43.066 I 49297 49297 [main.cpp:main:169] iter0: loss = 30.126541, w = 0.5257687, b = 0.99326295
JinboBook 2025-09-29 21:23:43.066 I 49297 49297 [main.cpp:main:169] iter1: loss = 18.83971, w = 0.9118613, b = 1.7899978
JinboBook 2025-09-29 21:23:43.066 I 49297 49297 [main.cpp:main:169] iter2: loss = 11.827219, w = 1.1956564, b = 2.4289458
JinboBook 2025-09-29 21:23:43.067 I 49297 49297 [main.cpp:main:169] iter3: loss = 7.450261, w = 1.4044737, b = 2.941251
JinboBook 2025-09-29 21:23:43.067 I 49297 49297 [main.cpp:main:169] iter4: loss = 4.7073665, w = 1.5582924, b = 3.351939
JinboBook 2025-09-29 21:23:43.067 I 49297 49297 [main.cpp:main:169] iter5: loss = 2.9825616, w = 1.6717329, b = 3.6811109
......
JinboBook 2025-09-29 21:23:43.808 I 49297 49297 [main.cpp:main:169] iter193: loss = 0.009776835, w = 1.9978435, b = 5.006678
JinboBook 2025-09-29 21:23:43.817 I 49297 49297 [main.cpp:main:169] iter194: loss = 0.009776835, w = 1.9978435, b = 5.006678
JinboBook 2025-09-29 21:23:43.826 I 49297 49297 [main.cpp:main:169] iter195: loss = 0.009776835, w = 1.9978435, b = 5.006678
JinboBook 2025-09-29 21:23:43.835 I 49297 49297 [main.cpp:main:169] iter196: loss = 0.009776835, w = 1.9978435, b = 5.006678
JinboBook 2025-09-29 21:23:43.845 I 49297 49297 [main.cpp:main:169] iter197: loss = 0.009776835, w = 1.9978435, b = 5.006678
JinboBook 2025-09-29 21:23:43.854 I 49297 49297 [main.cpp:main:169] iter198: loss = 0.009776835, w = 1.9978435, b = 5.006678
JinboBook 2025-09-29 21:23:43.864 I 49297 49297 [main.cpp:main:169] iter199: loss = 0.009776835, w = 1.9978435, b = 5.006678

```

## ⚡ 性能优化

### 后端选择

```cpp
// 使用CPU后端
af::setBackend(AF_BACKEND_CPU);

// 使用CUDA后端（需要NVIDIA GPU）
af::setBackend(AF_BACKEND_CUDA);

// 使用OpenCL后端
af::setBackend(AF_BACKEND_OPENCL);
```

### 环境变量设置

```bash
# 设置ArrayFire路径
export ARRAYFIRE_PATH=/opt/arrayfire
export LD_LIBRARY_PATH=${ARRAYFIRE_PATH}/lib64:$LD_LIBRARY_PATH

# 设置计算后端
export AF_BACKEND=CPU    # 或 CUDA, OPENCL
```

## ❓ 常见问题

### Q: 编译时找不到ArrayFire头文件
A: 确保设置了正确的 `ARRAYFIRE_PATH` 环境变量，并运行 `source setEnv.sh`

### Q: 运行时出现库文件找不到的错误
A: 确保 `LD_LIBRARY_PATH` 包含了ArrayFire的lib64目录

### Q: 如何添加新的算子？
A: 参考现有算子的实现，继承 `Operator` 类并实现 `forward` 和 `backward` 方法
