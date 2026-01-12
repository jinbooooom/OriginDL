# OriginDL API 文档

OriginDL 是一个C++深度学习框架，提供了类似PyTorch的API接口。本文档介绍了主要的API使用方法。

## 目录

- [张量创建](#张量创建)
- [张量属性](#张量属性)
- [张量操作](#张量操作)
- [数学运算](#数学运算)
- [调试工具](#调试工具)
- [神经网络模块](#神经网络模块)
- [CUDA 支持](#cuda-支持)
- [当前实现限制](#当前实现限制)

---

## 张量创建

### Tensor::zeros

```cpp
static Tensor zeros(const Shape &shape, const TensorOptions &options = TensorOptions())
```

返回一个全为0的张量，形状由`shape`参数定义。

**参数:**
- `shape` (Shape) – 定义输出张量的形状
- `options` (TensorOptions, optional) – 张量选项，包括数据类型、设备等

**返回值:** Tensor – 全为0的张量

**例子:**
```cpp
// 创建2x3的全零张量
auto t1 = Tensor::zeros({2, 3});
// t1.print() 输出:
// [[0, 0, 0],
//  [0, 0, 0]]
//  OriginMat(shape={2, 3}, dtype=float32, device=cpu)

// 指定数据类型为double
auto t2 = Tensor::zeros({3, 2}, TensorOptions().dtype(DataType::kFloat64));
// t2.print() 输出:
// [[0, 0],
//  [0, 0],
//  [0, 0]]
//  OriginMat(shape={3, 2}, dtype=float64, device=cpu)
```

### Tensor::ones

```cpp
static Tensor ones(const Shape &shape, const TensorOptions &options = TensorOptions())
```

返回一个全为1的张量，形状由`shape`参数定义。

**参数:**
- `shape` (Shape) – 定义输出张量的形状
- `options` (TensorOptions, optional) – 张量选项

**返回值:** Tensor – 全为1的张量

**例子:**
```cpp
// 创建2x3的全1张量
auto t1 = Tensor::ones({2, 3});
// t1.print() 输出:
// [[1, 1, 1],
//  [1, 1, 1]]
//  OriginMat(shape={2, 3}, dtype=float32, device=cpu)

// 创建5维向量
auto t2 = Tensor::ones({5});
// t2.print() 输出:
// [1, 1, 1, 1, 1]
//  OriginMat(shape={5}, dtype=float32, device=cpu)
```

### Tensor::randn

```cpp
static Tensor randn(const Shape &shape, const TensorOptions &options = TensorOptions())
```

返回一个随机张量，元素从标准正态分布中采样。

**参数:**
- `shape` (Shape) – 定义输出张量的形状
- `options` (TensorOptions, optional) – 张量选项

**返回值:** Tensor – 随机张量

**例子:**
```cpp
// 创建3x3的随机张量
auto t = Tensor::randn({3, 3});
// t.print() 输出:
// [[0.123, -0.456, 0.789],
//  [-0.234, 0.567, -0.890],
//  [0.345, -0.678, 0.901]]
//  OriginMat(shape={3, 3}, dtype=float32, device=cpu)
```

### Tensor::full

```cpp
static Tensor full(const Shape &shape, const Scalar &value, const TensorOptions &options = TensorOptions())
```

返回一个用指定值填充的张量。

**参数:**
- `shape` (Shape) – 定义输出张量的形状
- `value` (Scalar) – 填充值，支持多种数值类型
- `options` (TensorOptions, optional) – 张量选项

**返回值:** Tensor – 用指定值填充的张量

**例子:**
```cpp
// 创建用2.5f填充的2x2张量
auto t1 = Tensor::full({2, 2}, 2.5f);
// t1.print() 输出:
// [[2.5, 2.5],
//  [2.5, 2.5]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)

// 创建用-1.0f填充的3x1张量
auto t2 = Tensor::full({3, 1}, -1.0f);
// t2.print() 输出:
// [[-1],
//  [-1],
//  [-1]]
//  OriginMat(shape={3, 1}, dtype=float32, device=cpu)
```

### Tensor::from_blob

```cpp
static Tensor from_blob(void *data, const Shape &shape, const TensorOptions &options = TensorOptions())
```

从现有的内存数据创建张量。

**参数:**
- `data` (void*) – 指向数据的指针
- `shape` (Shape) – 张量形状
- `options` (TensorOptions, optional) – 张量选项

**返回值:** Tensor – 从数据创建的张量

**例子:**
```cpp
// 从数组创建张量
float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
auto t = Tensor::from_blob(data, {2, 2});
// t.print() 输出:
// [[1, 2],
//  [3, 4]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

### 张量构造函数

#### 从向量创建

```cpp
template <typename T>
Tensor(const std::vector<T> &data, const Shape &shape)
template <typename T>
Tensor(const std::vector<T> &data, const Shape &shape, DataType dtype)
template <typename T>
Tensor(const std::vector<T> &data, const Shape &shape, const TensorOptions &options)
```

从向量数据创建张量。

**参数:**
- `data` (std::vector<T>) – 向量数据
- `shape` (Shape) – 张量形状
- `dtype` (DataType, optional) – 指定数据类型，如果不指定则自动推断
- `options` (TensorOptions, optional) – 张量选项，可指定数据类型、设备等

**例子:**
```cpp
// 自动推断类型
std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
auto t1 = Tensor(data, {2, 2});
// t1.print() 输出:
// [[1, 2],
//  [3, 4]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)

// 指定数据类型
auto t2 = Tensor(data, {2, 2}, DataType::kFloat64);

// 使用TensorOptions指定完整选项
auto t3 = Tensor(data, {2, 2}, dtype(DataType::kFloat64).device(DeviceType::kCUDA));
```

#### 从初始化列表创建

```cpp
template <typename T>
Tensor(std::initializer_list<T> data, const Shape &shape)
template <typename T>
Tensor(std::initializer_list<T> data, const Shape &shape, DataType dtype)
template <typename T>
Tensor(std::initializer_list<T> data, const Shape &shape, const TensorOptions &options)
```

从初始化列表创建张量。

**参数:**
- `data` (std::initializer_list<T>) – 初始化列表数据
- `shape` (Shape) – 张量形状
- `dtype` (DataType, optional) – 指定数据类型
- `options` (TensorOptions, optional) – 张量选项

**例子:**
```cpp
// 自动推断类型
auto t1 = Tensor({1, 2, 3, 4}, {2, 2});
// t1.print() 输出:
// [[1, 2],
//  [3, 4]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)

// 指定数据类型
auto t2 = Tensor({1, 2, 3, 4}, {2, 2}, DataType::kInt32);

// 使用TensorOptions
auto t3 = Tensor({1, 2, 3, 4}, {2, 2}, TensorOptions().dtype(DataType::kFloat64));
```

#### 标量填充

```cpp
template <typename T>
Tensor(T scalar, const Shape &shape)
template <typename T>
Tensor(T scalar, const Shape &shape, DataType dtype)
template <typename T>
Tensor(T scalar, const Shape &shape, const TensorOptions &options)
```

用标量值填充整个张量。

**参数:**
- `scalar` (T) – 标量值
- `shape` (Shape) – 张量形状
- `dtype` (DataType, optional) – 指定数据类型
- `options` (TensorOptions, optional) – 张量选项

**例子:**
```cpp
// 自动推断类型
auto t1 = Tensor(5.0, {3, 3});
// t1.print() 输出:
// [[5, 5, 5],
//  [5, 5, 5],
//  [5, 5, 5]]
//  OriginMat(shape={3, 3}, dtype=float32, device=cpu)

// 指定数据类型
auto t2 = Tensor(5.0, {3, 3}, DataType::kFloat64);

// 使用TensorOptions
auto t3 = Tensor(5.0, {3, 3}, TensorOptions().dtype(DataType::kFloat64).device(DeviceType::kCUDA));
```

### TensorOptions 配置

TensorOptions 提供了灵活的配置选项，支持链式调用。

#### 数据类型设置

```cpp
TensorOptions &dtype(DataType dtype)
TensorOptions &dtype(const std::string &dtype_str)
TensorOptions &dtype(const char *dtype_str)
```

**例子:**
```cpp
// 设置数据类型为double
auto t = Tensor::zeros({2, 2}, dtype(DataType::kFloat64));

// 使用字符串设置类型
auto t2 = Tensor::ones({3, 3}, dtype("float64"));

// 使用C字符串设置类型
auto t3 = Tensor::zeros({2, 2}, dtype("float32"));
```

#### 设备设置

```cpp
TensorOptions &device(Device device)
TensorOptions &device(DeviceType device_type, int index = 0)
TensorOptions &device(const std::string &device_str)
TensorOptions &device(const char *device_str)
```

**例子:**
```cpp
// 设置设备为CUDA
auto options = TensorOptions().device(DeviceType::kCUDA, 0);
auto t = Tensor::randn({2, 2}, options);

// 使用字符串设置设备
auto options2 = TensorOptions().device("cuda:0");
auto t2 = Tensor::zeros({3, 3}, options2);

// 使用C字符串设置设备
auto options3 = TensorOptions().device("cpu");
auto t3 = Tensor::ones({2, 2}, options3);
```

#### 链式配置

```cpp
TensorOptions &requires_grad(bool requires_grad)
```

**例子:**
```cpp
// 链式配置多个选项
auto options = TensorOptions()
    .dtype(DataType::kFloat32)
    .device(DeviceType::kCPU)
    .requires_grad(true);
auto t = Tensor::ones({2, 2}, options);
```

#### 便捷函数

```cpp
TensorOptions dtype(DataType dtype)
TensorOptions dtype(const std::string &dtype_str)
TensorOptions dtype(const char *dtype_str)
TensorOptions device(Device device)
TensorOptions device(DeviceType device_type, int index = 0)
TensorOptions device(const std::string &device_str)
TensorOptions device(const char *device_str)
TensorOptions requires_grad(bool requires_grad = true)
```

**例子:**
```cpp
// 快速创建配置
auto t1 = Tensor::zeros({2, 2}, dtype(DataType::kFloat64));
auto t2 = Tensor::ones({3, 3}, device(DeviceType::kCUDA));
auto t3 = Tensor::randn({2, 2}, requires_grad(false));

// 使用字符串便捷函数
auto t4 = Tensor::zeros({2, 2}, dtype("float64"));
auto t5 = Tensor::ones({3, 3}, device("cuda:0"));
```

---

## 张量属性

### shape

```cpp
Shape shape() const
```

返回张量的形状。

**返回值:** Shape – 张量形状

**例子:**
```cpp
auto t = Tensor::ones({3, 4, 5});
auto s = t.shape();  // Shape{3, 4, 5}
```

### ndim

```cpp
size_t ndim() const
```

返回张量的维度数。

**返回值:** size_t – 维度数

**例子:**
```cpp
auto t = Tensor::ones({3, 4, 5});
auto d = t.ndim();  // 3
```

### elements / numel

```cpp
size_t elements() const
size_t numel() const
```

返回张量中元素的总数。

**返回值:** size_t – 元素总数

**例子:**
```cpp
auto t = Tensor::ones({3, 4, 5});
auto n = t.elements();  // 60
auto n2 = t.numel();    // 60 (与elements()相同)
```

### dtype

```cpp
DataType dtype() const
```

返回张量的数据类型。

**返回值:** DataType – 数据类型

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat64));
auto dt = t.dtype();  // DataType::kFloat64
```

### device

```cpp
Device device() const
```

返回张量所在的设备。

**返回值:** Device – 设备信息

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, device(DeviceType::kCUDA));
auto dev = t.device();  // Device(DeviceType::kCUDA, 0)
```

### element_size

```cpp
size_t element_size() const
```

返回单个元素的字节大小。

**返回值:** size_t – 元素字节大小

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat32));
auto size = t.element_size();  // 4 (float32占4字节)
```

### nbytes

```cpp
size_t nbytes() const
```

返回张量占用的总字节数。

**返回值:** size_t – 总字节数

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat32));
auto bytes = t.nbytes();  // 16 (4个元素 × 4字节)
```

### 梯度相关

#### grad

```cpp
Tensor grad() const
```

获取张量的梯度。如果未执行 backward()，将返回一个全零张量。

**返回值:** Tensor – 梯度张量

**注意:**
- 如果梯度未初始化（未执行 `backward()`），将返回一个全零张量，形状和设备与原始张量相同
- 这与 PyTorch 的行为不同：PyTorch 在梯度未初始化时返回 `None`
- 返回的梯度类型始终为 `float32`，即使原始张量是其他类型（见[当前实现限制](#当前实现限制)）

**例子:**
```cpp
auto x = Tensor::ones({2, 2}, requires_grad(true));

// 未执行 backward() 时，grad() 返回全零张量
auto grad_before = x.grad();  // 返回全零张量，形状为 {2, 2}

auto y = x * x;
y.backward();
auto g = x.grad();  // 获取x的梯度
// g.print() 输出梯度值
```

#### backward

```cpp
void backward()
```

执行反向传播，计算计算图中所有张量的梯度。

**例子:**
```cpp
auto x = Tensor::ones({2, 2});
auto y = x * x;
y.backward();  // 执行反向传播
// 此时x.grad()包含了y对x的梯度
```

#### clear_grad

```cpp
void clear_grad()
```

清除张量的梯度。

**例子:**
```cpp
auto x = Tensor::ones({2, 2});
auto y = x * x;
y.backward();
x.clear_grad();  // 清除x的梯度
// 之后x.grad()将返回全零张量
```

#### detach

```cpp
Tensor detach() const
```

断开张量与计算图的连接，创建一个不参与梯度计算的新张量。类似于 PyTorch 的 `detach()` 方法。

**返回值:** Tensor – 新的张量，与原始张量共享数据但不参与梯度计算

**注意:**
- 返回的新张量不包含 `creator_` 和 `grad_`，因此不会参与反向传播
- 调用 `detach()` 会递归清理整个计算图，断开所有相关的 Operator 引用，帮助释放 GPU 内存
- 这对于解决循环引用导致的内存泄漏问题非常有用
- 在训练循环中，可以在 `backward()` 和 `step()` 之后调用 `detach()` 来显式释放计算图内存

**例子:**
```cpp
auto x = Tensor::ones({2, 2}, requires_grad(true));
auto y = x * x;
auto loss = sum(y);

// 反向传播
loss.backward();

// 断开计算图，释放内存
auto detached_loss = loss.detach();
// detached_loss 与 loss 共享数据，但不参与梯度计算
// 调用 detach() 后，整个计算图（包括所有中间 tensor）都可以被释放

// 在训练循环中的典型用法
for (int i = 0; i < epochs; ++i) {
    auto y = model(x);
    auto loss = compute_loss(y, target);
    
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    
    // 显式断开计算图，释放 GPU 内存
    loss.detach();
    y.detach();
}
```

#### clone

```cpp
Tensor clone() const
```

克隆张量（深拷贝数据，保留计算图连接）。类似于 PyTorch 的 `clone()` 方法。

**返回值:** Tensor – 新的张量，与原始张量数据独立但保留计算图连接

**注意:**
- 深拷贝 `data_`（创建独立的数据副本）
- 不复制 `grad_`（初始化为 `nullptr`，需要重新计算梯度）
- 复制 `creator_` 和 `generation_`（保留计算图连接，仍可参与梯度计算）
- 如果需要完全独立（断开计算图），使用 `clone().detach()`

**例子:**
```cpp
auto x = Tensor::ones({2, 2});
auto y = x * x;
auto loss = sum(y);

// 克隆tensor，保留计算图连接
auto cloned_loss = loss.clone();
// cloned_loss 与 loss 数据独立，但仍可参与梯度计算

// 反向传播（对原始loss）
loss.backward();
// 此时 loss 有梯度，但 cloned_loss 没有梯度（需要重新计算）

// 如果需要完全独立的tensor（断开计算图）
auto independent_loss = loss.clone().detach();
// independent_loss 数据独立，且不参与梯度计算

// 典型用法：在需要修改tensor数据但不想影响原始tensor时
auto x_cloned = x.clone();
x_cloned.data_ptr<float>()[0] = 999.0f;  // 修改克隆的tensor
// x 的值不受影响，因为数据是独立的
```

---

## 张量操作

### reshape

```cpp
Tensor reshape(const Shape &shape) const
```

重塑张量的形状，保持元素总数不变。

**参数:**
- `shape` (Shape) – 新的形状

**返回值:** Tensor – 重塑后的张量

**例子:**
```cpp
auto t = Tensor::ones({2, 6});
auto reshaped = t.reshape({3, 4});
// 原始: 2x6
// 重塑后: 3x4
```

### transpose

```cpp
Tensor transpose() const
```

转置张量（交换最后两个维度）。

**返回值:** Tensor – 转置后的张量

**例子:**
```cpp
auto t = Tensor::ones({2, 3});
auto transposed = t.transpose();
// 原始: 2x3
// 转置后: 3x2
```

### to (类型转换)

```cpp
Tensor to(DataType target_type) const
Tensor to(Device device) const
Tensor to(const TensorOptions &options) const
```

转换张量的数据类型或设备。

**参数:**
- `target_type` (DataType) – 目标数据类型
- `device` (Device) – 目标设备
- `options` (TensorOptions) – 目标选项

**返回值:** Tensor – 转换后的新张量

**注意:**
- `to()` 方法总是返回一个新的张量，不会修改原张量（非原地操作）
- 如果目标类型或设备与当前张量相同，仍会创建一个新的张量对象
- 原张量的数据不会被修改，可以安全地继续使用

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat32));

// 转换数据类型（返回新张量，t保持不变）
auto t_float64 = t.to(DataType::kFloat64);
// t 仍然是 float32 类型

// 转换设备（返回新张量）
auto t_cuda = t.to(Device(DeviceType::kCUDA));
// t 仍然在 CPU 上

// 同时转换类型和设备
auto options = TensorOptions().dtype(DataType::kFloat64).device(DeviceType::kCUDA);
auto t_both = t.to(options);
```

### item

```cpp
template <typename T>
T item() const
```

获取标量张量的值。

**参数:**
- `T` – 返回值的类型，必须与张量的数据类型兼容

**返回值:** T – 标量值

**注意:**
- `item()` 只能用于标量张量（元素数量为1的张量）
- 如果张量不是标量（元素数量大于1），调用 `item()` 会抛出 `RuntimeError` 异常
- 对于0维张量（形状为 `{}`）或1维单元素张量（形状为 `{1}`），都可以使用 `item()`

**例子:**
```cpp
// 标量张量（0维，形状为 {}）
auto t1 = Tensor::full({}, 3.14f);  // 或者 Tensor::full(Shape{}, 3.14f)
float value1 = t1.item<float>();  // 3.14

// 单元素张量（1维，形状为 {1}）
auto t2 = Tensor({3.14f}, {1});
float value2 = t2.item<float>();  // 3.14

// 错误示例：非标量张量
auto t3 = Tensor::ones({2, 2});
// float value3 = t3.item<float>();  // 会抛出异常：tensor has 4 elements
```

### data_ptr

```cpp
template <typename T>
T *data_ptr()
```

获取张量数据的原始指针。

**参数:**
- `T` – 指针类型，必须与张量的数据类型匹配

**返回值:** T* – 指向张量数据的原始指针

**注意:**
- **内存管理**: 返回的指针指向张量内部管理的存储空间，指针的生命周期与张量对象绑定。当张量对象被销毁时，指针将失效
- **线程安全**: `data_ptr()` 本身不是线程安全的。如果多个线程同时访问同一个张量的数据指针，需要自行实现同步机制
- **数据修改**: 通过返回的指针修改数据会直接影响张量的值，这可能会影响计算图和梯度计算。建议仅在必要时使用，并确保了解其影响
- **设备限制**: 对于 CUDA 张量，返回的指针指向 GPU 内存，在 CPU 代码中直接访问可能导致未定义行为。需要先将数据复制到 CPU
- **类型匹配**: 模板参数 `T` 必须与张量的实际数据类型匹配，否则可能导致未定义行为

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat32));
float* ptr = t.data_ptr<float>();

// 通过指针访问和修改数据
for (size_t i = 0; i < t.elements(); ++i) {
    ptr[i] = i * 0.1f;
}

// 注意：修改指针指向的数据会影响原张量
t.print();  // 会显示修改后的值
```

### to_vector

```cpp
template <typename T>
std::vector<T> to_vector() const
```

将张量转换为向量。

**返回值:** std::vector<T> – 向量数据

**例子:**
```cpp
auto t = Tensor::ones({2, 2});
auto vec = t.to_vector<float>();
// vec: {1.0, 1.0, 1.0, 1.0}
```

---

## 数学运算

### 基础算术运算

#### 加法

```cpp
Tensor operator+(const Tensor &lhs, const Tensor &rhs)
Tensor add(const Tensor &lhs, const Tensor &rhs)
```

张量加法，支持广播。

**例子:**
```cpp
auto a = Tensor::ones({2, 2});
auto b = Tensor::full({2, 2}, 2.0);
auto c = a + b;  // 或者 add(a, b)
// c.print() 输出:
// [[3, 3],
//  [3, 3]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

#### 减法

```cpp
Tensor operator-(const Tensor &lhs, const Tensor &rhs)
Tensor sub(const Tensor &lhs, const Tensor &rhs)
```

张量减法。

**例子:**
```cpp
auto a = Tensor::full({2, 2}, 5.0);
auto b = Tensor::ones({2, 2});
auto c = a - b;  // 或者 sub(a, b)
// c.print() 输出:
// [[4, 4],
//  [4, 4]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

#### 乘法

```cpp
Tensor operator*(const Tensor &lhs, const Tensor &rhs)
Tensor mul(const Tensor &lhs, const Tensor &rhs)
```

张量乘法（逐元素）。

**例子:**
```cpp
auto a = Tensor::full({2, 2}, 3.0);
auto b = Tensor::full({2, 2}, 2.0);
auto c = a * b;  // 或者 mul(a, b)
// c.print() 输出:
// [[6, 6],
//  [6, 6]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

#### 除法

```cpp
Tensor operator/(const Tensor &lhs, const Tensor &rhs)
Tensor div(const Tensor &lhs, const Tensor &rhs)
```

张量除法。

**例子:**
```cpp
auto a = Tensor::full({2, 2}, 8.0);
auto b = Tensor::full({2, 2}, 2.0);
auto c = a / b;  // 或者 div(a, b)
// c.print() 输出:
// [[4, 4],
//  [4, 4]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

#### 取负

```cpp
Tensor operator-(const Tensor &x)
Tensor neg(const Tensor &x)
```

张量取负。

**例子:**
```cpp
auto a = Tensor::ones({2, 2});
auto b = -a;  // 或者 neg(a)
// b.print() 输出:
// [[-1, -1],
//  [-1, -1]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

### 标量运算

支持张量与标量的运算：

```cpp
template <typename T>
Tensor operator+(const Tensor &lhs, T rhs)
template <typename T>
Tensor operator+(T lhs, const Tensor &rhs)
// 减法、乘法、除法类似
```

**类型提升规则:**

当张量与标量进行运算时，框架会自动进行类型提升，确保运算结果使用精度更高的类型。类型提升的优先级规则如下（从高到低）：

1. **浮点类型**: `double` (float64) > `float` (float32)
2. **整数类型**: `int64` > `int32` > `int16` > `int8`
3. **无符号整数**: `uint64` > `uint32` > `uint16` > `uint8`
4. **布尔类型**: `bool`

**提升规则说明:**
- 如果张量和标量类型相同，结果类型保持不变
- 如果类型不同，结果类型为两者中优先级更高的类型
- 浮点类型优先级高于整数类型（例如：`float32` 张量 + `int32` 标量 → `float32` 结果）
- 整数类型之间按精度提升（例如：`int32` 张量 + `int64` 标量 → `int64` 结果）

**例子:**
```cpp
// 相同类型，不提升
auto a1 = Tensor::ones({2, 2}, dtype(DataType::kFloat32));
auto b1 = a1 + 2.0f;  // 结果类型: float32

// 类型提升：float32 + double → double
auto a2 = Tensor::ones({2, 2}, dtype(DataType::kFloat32));
auto b2 = a2 + 2.0;   // 结果类型: float64 (double)

// 类型提升：int32 + float32 → float32
auto a3 = Tensor({1, 2, 3, 4}, {2, 2}, DataType::kInt32);
auto b3 = a3 + 2.5f;  // 结果类型: float32

// 类型提升：int32 + int64 → int64
auto a4 = Tensor({1, 2, 3, 4}, {2, 2}, DataType::kInt32);
auto b4 = a4 + 2L;    // 结果类型: int64
```

### 数学函数

#### square

```cpp
Tensor square(const Tensor &x)
```

计算张量的平方。

**例子:**
```cpp
auto a = Tensor({1, 2, 3, 4}, {2, 2});
auto b = square(a);
// a.print() 输出:
// [[1, 2],
//  [3, 4]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
// b.print() 输出:
// [[1, 4],
//  [9, 16]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)
```

#### pow

```cpp
Tensor pow(const Tensor &base, const Scalar &exponent)
Tensor operator^(const Tensor &base, const Scalar &exponent)
```

计算张量的幂。

**参数:**
- `base` (Tensor) – 底数张量
- `exponent` (Scalar) – 指数值

**返回值:** Tensor – 幂运算结果

**注意:**
- **负数底数与非整数指数**: 当底数为负数且指数为非整数时，结果为 `NaN`（Not a Number）。这是因为负数的非整数次幂在实数域中无定义，底层使用 `std::pow` 函数会返回 `NaN`。例如：`pow(-2.0, 2.5)` 会产生 `NaN`
- **负数底数与整数指数**: 当底数为负数但指数为整数时，运算正常进行。例如：`pow(-2.0, 2)` 结果为 `4.0`，`pow(-2.0, 3)` 结果为 `-8.0`
- **零的负指数**: 当底数为0且指数为负数时，结果可能为 `inf`（无穷大）或抛出异常
- **类型提升**: 如果底数张量和指数的类型不同，会按照类型提升规则自动提升到更高精度的类型

**例子:**
```cpp
auto a = Tensor({2, 3, 4}, {1, 3});
auto b = pow(a, 2);    // 使用函数形式
auto c = a ^ 2;        // 使用操作符形式（等价于pow(a, 2)）
// a.print() 输出:
// [[2, 3, 4]]
//  OriginMat(shape={1, 3}, dtype=float32, device=cpu)
// b.print() 输出:
// [[4, 9, 16]]
//  OriginMat(shape={1, 3}, dtype=float32, device=cpu)

// 支持不同数值类型的指数
auto d = pow(a, 3.0);   // 浮点数指数
auto e = a ^ 2.5;       // 使用操作符形式

// 负数底数示例
auto neg = Tensor({-2.0, -3.0}, {1, 2});
auto pos_int_pow = pow(neg, 2);     // 结果: [4.0, 9.0]（正常）
auto neg_int_pow = pow(neg, 3);     // 结果: [-8.0, -27.0]（正常）
auto non_int_pow = pow(neg, 2.5);   // 结果: [NaN, NaN]（负数非整数次幂产生NaN）
```

#### exp

```cpp
Tensor exp(const Tensor &x)
```

计算张量的指数函数。

**例子:**
```cpp
auto a = Tensor({0, 1, 2}, {1, 3});
auto b = exp(a);
// a.print() 输出:
// [[0, 1, 2]]
//  OriginMat(shape={1, 3}, dtype=float32, device=cpu)
// b.print() 输出:
// [[1, 2.718, 7.389]]
//  OriginMat(shape={1, 3}, dtype=float32, device=cpu)
```

### 形状操作

#### reshape (函数版本)

```cpp
Tensor reshape(const Tensor &x, const Shape &shape)
```

重塑张量形状。

**例子:**
```cpp
auto a = Tensor::ones({2, 6});
auto b = reshape(a, {3, 4});
```

#### transpose (函数版本)

```cpp
Tensor transpose(const Tensor &x)
```

转置张量。

**例子:**
```cpp
auto a = Tensor::ones({2, 3});
auto b = transpose(a);
// a: 2x3
// b: 3x2
```

#### broadcast_to

```cpp
Tensor broadcast_to(const Tensor &x, const Shape &shape)
```

将张量广播到指定形状。

**例子:**
```cpp
auto a = Tensor({1, 2}, {1, 2});
auto b = broadcast_to(a, {3, 2});
// a.print() 输出:
// [[1, 2]]
//  OriginMat(shape={1, 2}, dtype=float32, device=cpu)
// b.print() 输出:
// [[1, 2],
//  [1, 2],
//  [1, 2]]
//  OriginMat(shape={3, 2}, dtype=float32, device=cpu)
```

#### sum_to

```cpp
Tensor sum_to(const Tensor &x, const Shape &shape)
```

将张量求和到指定形状。

**例子:**
```cpp
auto a = Tensor::ones({3, 2});
auto b = sum_to(a, {1, 2});
// 将3x2的张量求和到1x2
```

### 归约操作

#### sum

```cpp
Tensor sum(const Tensor &x, int axis = -1)
```

对张量求和。

**参数:**
- `x` (Tensor) – 输入张量
- `axis` (int, optional) – 求和的轴，-1表示所有元素

**例子:**
```cpp
auto a = Tensor({1, 2, 3, 4}, {2, 2});
auto b = sum(a);      // 所有元素求和: 10
auto c = sum(a, 0);   // 按第0轴求和
auto d = sum(a, 1);   // 按第1轴求和
```

### 矩阵运算

#### mat_mul

```cpp
Tensor mat_mul(const Tensor &x, const Tensor &w)
```

矩阵乘法（张量乘法）。

**参数:**
- `x` (Tensor) – 第一个张量，形状应为 `[..., m, n]`
- `w` (Tensor) – 第二个张量，形状应为 `[..., n, p]`

**返回值:** Tensor – 矩阵乘法结果张量，形状为 `[..., m, p]`

**注意:**
- 这是真正的矩阵乘法（不是逐元素乘法），对应数学中的矩阵乘法运算
- **当前实现限制**：OriginDL 当前版本仅支持以下两种形式：
  - **2D x 2D**: `{m, k} x {k, n}` → `{m, n}`（标准矩阵乘法）
  - **3D x 2D**: `{batch, m, k} x {k, n}` → `{batch, m, n}`（批量矩阵乘法）
- 第一个张量的最后一个维度必须与第二个张量的第一个维度相同（`k` 必须匹配）
- **与 PyTorch 的差异**：PyTorch 的 `torch.matmul` 支持更广泛的形状组合，包括：
  - `{batch, m, k} x {batch, k, n}` → `{batch, m, n}`（两个3D张量的批量矩阵乘法）
  - `{m, k} x {batch, k, n}` → `{batch, m, n}`（2D x 3D，自动广播）
  - `{batch, m, k} x {k, n}` → `{batch, m, n}`（3D x 2D，已支持）
  - 更高维度的批量矩阵乘法
- 当前实现不支持 3D x 3D、2D x 3D 等组合，这些功能可能在未来的版本中添加

**例子:**
```cpp
// 2D张量矩阵乘法
auto a = Tensor::ones({2, 3});
auto b = Tensor::ones({3, 4});
auto c = mat_mul(a, b);
// 结果: 2x4的张量

// 批量矩阵乘法（3D x 2D）
// 对3D张量的最后两个维度进行矩阵乘法，第一个维度作为批量维度
auto batch_a = Tensor::ones({10, 2, 3});  // 形状: {10, 2, 3}，10个2x3的张量
auto b = Tensor::ones({3, 4});            // 形状: {3, 4}，共享的权重张量
auto batch_c = mat_mul(batch_a, b);       // 结果: {10, 2, 4}，10个2x4的张量
```

---

## 调试工具

### print

```cpp
void print(const std::string &desc = "") const
```

打印张量的内容。

**参数:**
- `desc` (std::string, optional) – 描述信息

**例子:**
```cpp
auto t = Tensor::ones({2, 2});
t.print("我的张量");
// 输出:
// 我的张量:
// [[1, 1],
//  [1, 1]]
//  OriginMat(shape={2, 2}, dtype=float32, device=cpu)

// 高维张量使用分块显示格式
auto t3d = Tensor::ones({2, 3, 2});
t3d.print("3D张量");
// 输出:
// 3D张量:
// (0,.,.) = 
//      1       1
//      1       1
//      1       1
// (1,.,.) = 
//      1       1
//      1       1
//      1       1
//  OriginMat(shape={2, 3, 2}, dtype=float32, device=cpu)

// 大张量会自动省略显示
auto large_t = Tensor::ones({100, 25});
large_t.print("大张量");
// 输出:
// 大张量:
// [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, ...],
//  [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, ...],
//  ...]
//  OriginMat(shape={100, 25}, dtype=float32, device=cpu)
```

### backend_type

```cpp
int backend_type() const
```

获取张量使用的后端类型。

**返回值:** int – 后端类型标识

**例子:**
```cpp
auto t = Tensor::ones({2, 2});
int backend = t.backend_type();
// 0: Origin后端
// 1: Torch后端
```

---

## 神经网络模块

OriginDL 提供了神经网络模块，支持构建和训练深度学习模型。模块设计参考 PyTorch，提供了类似的 API 接口。

### Module 基类

所有神经网络模块都继承自 `Module` 基类，提供了参数管理、训练模式切换、设备迁移等核心功能。

#### forward

```cpp
virtual Tensor forward(const Tensor &input)
```

执行前向传播。这是所有模块必须实现的纯虚函数。

**参数:**
- `input` (Tensor) – 输入张量

**返回值:** Tensor – 输出张量

**例子:**
```cpp
Sequential model;
model.add(std::make_unique<Linear>(10, 5));
Tensor input = Tensor::randn({32, 10});
Tensor output = model.forward(input);
```

#### operator()

```cpp
Tensor operator()(const Tensor &input)
```

调用操作符，等价于 `forward()`。提供更简洁的调用方式。

**例子:**
```cpp
auto output = model(input);  // 等价于 model.forward(input)
```

#### train / eval

```cpp
void train(bool mode = true)
void eval()
```

设置模块的训练模式或评估模式。

**参数:**
- `mode` (bool, optional) – 训练模式标志，默认为 `true`

**注意:**
- `train()` 设置训练模式，启用 dropout、batch normalization 等训练时的行为
- `eval()` 设置评估模式，禁用训练时的特殊行为

**例子:**
```cpp
model.train();  // 设置为训练模式
// ... 训练代码 ...
model.eval();   // 设置为评估模式
// ... 评估代码 ...
```

#### to (设备迁移)

```cpp
virtual void to(Device device)
void to(const TensorOptions &options)
```

将模块及其所有参数迁移到指定设备。

**参数:**
- `device` (Device) – 目标设备
- `options` (TensorOptions) – 张量选项（包含设备和数据类型）

**例子:**
```cpp
// 迁移到 CUDA 设备
model.to(Device(DeviceType::kCUDA, 0));

// 使用 TensorOptions
model.to(TensorOptions().device(DeviceType::kCUDA).dtype(DataType::kFloat32));
```

#### zero_grad

```cpp
void zero_grad()
```

清除模块中所有参数的梯度。

**例子:**
```cpp
optimizer.zero_grad();  // 通常在优化器中调用，会自动清除模型参数梯度
```

#### parameters

```cpp
virtual std::vector<Parameter *> parameters()
```

获取模块中所有参数的列表。

**返回值:** std::vector<Parameter *> – 参数指针向量

**例子:**
```cpp
auto params = model.parameters();
for (auto *param : params) {
    // 访问参数
}
```

### Sequential 容器

`Sequential` 是一个顺序容器，用于按顺序组织多个模块。

#### 构造函数

```cpp
Sequential()
```

创建空的 Sequential 容器。

**例子:**
```cpp
Sequential model;
```

#### add

```cpp
void add(std::unique_ptr<Module> module)
```

向容器中添加模块。

**参数:**
- `module` (std::unique_ptr<Module>) – 要添加的模块

**例子:**
```cpp
Sequential model;
model.add(std::make_unique<Linear>(10, 5));
model.add(std::make_unique<Linear>(5, 1));
```

#### forward

```cpp
Tensor forward(const Tensor &input) override
```

按顺序执行所有模块的前向传播。

**例子:**
```cpp
Sequential model;
model.add(std::make_unique<Linear>(10, 5));
model.add(std::make_unique<Linear>(5, 1));

Tensor input = Tensor::randn({32, 10});
Tensor output = model(input);  // 依次通过两个 Linear 层
```

#### operator[]

```cpp
Module &operator[](size_t index)
const Module &operator[](size_t index) const
```

通过索引访问容器中的模块。

**参数:**
- `index` (size_t) – 模块索引

**返回值:** Module & – 模块引用

**例子:**
```cpp
Sequential model;
model.add(std::make_unique<Linear>(10, 5));
model.add(std::make_unique<Linear>(5, 1));

// 访问第一个模块
auto &first_layer = model[0];
auto &linear_layer = dynamic_cast<Linear &>(model[0]);
float w = linear_layer.weight()->item<float>();
```

#### size

```cpp
size_t size() const
```

获取容器中模块的数量。

**返回值:** size_t – 模块数量

**例子:**
```cpp
Sequential model;
model.add(std::make_unique<Linear>(10, 5));
std::cout << "Number of layers: " << model.size() << std::endl;  // 输出: 1
```

### Linear 层

`Linear` 是全连接层（线性层），实现 `y = x * W + b`。

#### 构造函数

```cpp
Linear(int in_features, int out_features, bool bias = true)
```

创建线性层。

**参数:**
- `in_features` (int) – 输入特征数
- `out_features` (int) – 输出特征数
- `bias` (bool, optional) – 是否使用偏置，默认为 `true`

**例子:**
```cpp
// 创建输入10维、输出5维的线性层，带偏置
auto linear = std::make_unique<Linear>(10, 5, true);

// 创建不带偏置的线性层
auto linear_no_bias = std::make_unique<Linear>(10, 5, false);
```

#### forward

```cpp
Tensor forward(const Tensor &input) override
```

执行线性变换：`output = input * weight^T + bias`

**参数:**
- `input` (Tensor) – 输入张量，形状应为 `[..., in_features]`

**返回值:** Tensor – 输出张量，形状为 `[..., out_features]`

**例子:**
```cpp
Linear linear(10, 5);
Tensor input = Tensor::randn({32, 10});
Tensor output = linear(input);  // 输出形状: {32, 5}
```

#### weight / bias

```cpp
Parameter *weight()
Parameter *bias()
```

访问权重和偏置参数。

**返回值:** Parameter * – 参数指针，`bias()` 在未使用偏置时返回 `nullptr`

**例子:**
```cpp
Linear linear(10, 5);
auto *w = linear.weight();
auto *b = linear.bias();

// 访问参数值
float w_val = w->item<float>();
if (b != nullptr) {
    float b_val = b->item<float>();
}
```

#### reset_parameters

```cpp
void reset_parameters()
```

重置参数，使用默认初始化策略重新初始化权重和偏置。

**例子:**
```cpp
Linear linear(10, 5);
linear.reset_parameters();  // 重新初始化参数
```

### Optimizer 优化器

`Optimizer` 是优化器基类，用于更新模型参数。

#### 构造函数

```cpp
explicit Optimizer(Module &target)
```

创建优化器。

**参数:**
- `target` (Module &) – 目标模块

**注意:** 通常不直接使用 `Optimizer`，而是使用其子类如 `SGD`。

#### step

```cpp
void step()
```

执行一步参数更新。

**例子:**
```cpp
SGD optimizer(model, 0.01f);
// ... 计算梯度 ...
optimizer.step();  // 更新参数
```

#### zero_grad

```cpp
void zero_grad()
```

清除所有参数的梯度。

**例子:**
```cpp
optimizer.zero_grad();  // 在每次迭代开始时调用
```

### SGD 优化器

`SGD` 实现随机梯度下降优化算法。

#### 构造函数

```cpp
SGD(Module &target, float lr, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false)
```

创建 SGD 优化器。

**参数:**
- `target` (Module &) – 目标模块
- `lr` (float) – 学习率
- `momentum` (float, optional) – 动量系数，默认为 0.0
- `weight_decay` (float, optional) – 权重衰减（L2正则化），默认为 0.0
- `nesterov` (bool, optional) – 是否使用 Nesterov 动量，默认为 false

**例子:**
```cpp
// 基础 SGD
SGD optimizer(model, 0.1f);

// 带动量的 SGD
SGD optimizer_momentum(model, 0.1f, 0.9f);

// 带权重衰减的 SGD
SGD optimizer_decay(model, 0.1f, 0.0f, 0.0001f);

// Nesterov 动量 SGD
SGD optimizer_nesterov(model, 0.1f, 0.9f, 0.0f, true);
```

#### step

```cpp
void step() override
```

执行一步 SGD 更新。

**例子:**
```cpp
SGD optimizer(model, 0.1f);

for (int i = 0; i < num_iterations; ++i) {
    optimizer.zero_grad();
    
    // 前向传播
    auto output = model(input);
    
    // 计算损失
    auto loss = compute_loss(output, target);
    
    // 反向传播
    loss.backward();
    
    // 更新参数
    optimizer.step();
}
```

### 完整训练示例

以下是一个完整的线性回归训练示例：

```cpp
#include "origin.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/sequential.h"
#include "origin/optim/sgd.h"

using namespace origin;

int main() {
    // 1. 创建训练数据
    size_t input_size = 100;
    auto x = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32));
    auto noise = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32)) * 0.1f;
    auto y = x * 2.0f + 5.0f + noise;  // y = 2x + 5 + noise

    // 2. 创建模型
    Sequential model;
    model.add(std::make_unique<Linear>(1, 1, true));  // 输入1维，输出1维，带偏置

    // 3. 创建优化器
    float learning_rate = 0.1f;
    SGD optimizer(model, learning_rate);

    // 4. 开始训练
    int iters = 200;
    model.train();  // 设置为训练模式

    for (int i = 0; i < iters; ++i) {
        optimizer.zero_grad();
        
        // 前向传播
        auto y_pred = model(x);
        
        // 计算损失（MSE）
        auto diff = y_pred - y;
        auto sum_result = sum(pow(diff, 2));
        auto elements = Tensor(diff.elements(), sum_result.shape(), DataType::kFloat32);
        auto loss = sum_result / elements;
        
        // 反向传播
        loss.backward();
        
        // 更新参数
        optimizer.step();
        
        // 打印训练进度
        if (i % 10 == 0 || i == iters - 1) {
            float loss_val = loss.item<float>();
            auto &linear_layer = dynamic_cast<Linear &>(model[0]);
            float w_val = linear_layer.weight()->item<float>();
            float b_val = linear_layer.bias()->item<float>();
            
            std::cout << "iter " << i << ": loss = " << loss_val 
                      << ", w = " << w_val << ", b = " << b_val << std::endl;
        }
    }
    
    return 0;
}
```

---

## CUDA 支持

OriginDL 提供了 CUDA 支持，允许在 GPU 上进行张量计算。使用 CUDA 功能需要：
1. 系统安装 NVIDIA CUDA Toolkit
2. 编译时启用 CUDA 支持（使用 `--cuda` 标志）
3. 运行时系统有可用的 CUDA 设备

### CUDA 异步执行策略

OriginDL 采用与 PyTorch 类似的异步执行策略，以最大化 GPU 利用率：

- **算子操作（异步）**：所有 CUDA 算子（如 `add`、`multiply`、`mat_mul` 等）在启动 kernel 后立即返回，不等待 GPU 完成计算。这允许 CPU 和 GPU 并行工作，提高整体性能。

- **关键同步点**：仅在以下关键位置进行同步（调用 `cudaDeviceSynchronize()`）：
  - `item()` 调用时：需要从 GPU 读取标量值到 CPU
  - `to_vector()` 调用时：需要从 GPU 复制数据到 CPU
  - `to_device()` 从 GPU 复制到 CPU 时：确保数据复制前所有 GPU 操作已完成
  - 数据复制操作（如 `clone()`）：确保复制的是最新的数据

这种设计允许：
- 多个算子操作可以流水线执行
- CPU 可以在 GPU 计算时继续执行其他任务
- 减少不必要的同步开销，提高性能

**注意**：由于 CUDA 操作的异步特性，如果需要在 CPU 上访问 GPU 张量的数据，必须通过 `item()` 或 `to_vector()` 等方法，这些方法会自动处理必要的同步。

### origin::cuda::is_available

```cpp
bool origin::cuda::is_available()
```

检查当前系统是否有可用的 CUDA 设备。这是 PyTorch 中 `torch.cuda.is_available()` 的对应 API。

**返回值:** bool – 如果 CUDA 可用返回 `true`，否则返回 `false`

**注意:** 
- 此函数需要在编译时启用 CUDA 支持（`WITH_CUDA` 宏定义）
- 如果未启用 CUDA 支持，包含 `origin.h` 时不会暴露 CUDA 命名空间

**例子:**
```cpp
#include "origin.h"

#ifdef WITH_CUDA
    // 检查 CUDA 是否可用
    if (origin::cuda::is_available()) {
        std::cout << "CUDA is available!" << std::endl;
        
        // 创建 CUDA 张量
        auto t = Tensor::ones({2, 2}, 
                              TensorOptions().device(DeviceType::kCUDA));
        t.print("CUDA Tensor");
    } else {
        std::cout << "CUDA is not available on this system." << std::endl;
    }
#else
    std::cout << "CUDA support is not compiled in." << std::endl;
#endif
```

### origin::cuda::device_count

```cpp
int origin::cuda::device_count()
```

返回系统中可用的 CUDA 设备数量。这是 PyTorch 中 `torch.cuda.device_count()` 的对应 API。

**返回值:** int – 可用的 CUDA 设备数量，如果没有可用设备则返回 0

**例子:**
```cpp
#include "origin.h"

#ifdef WITH_CUDA
    if (origin::cuda::is_available()) {
        int count = origin::cuda::device_count();
        std::cout << "Number of CUDA devices: " << count << std::endl;
        
        // 遍历所有设备
        for (int i = 0; i < count; ++i) {
            origin::cuda::set_device(i);
            std::cout << "Using device " << i << std::endl;
        }
    }
#endif
```

### origin::cuda::current_device

```cpp
int origin::cuda::current_device()
```

返回当前选定的 CUDA 设备索引。这是 PyTorch 中 `torch.cuda.current_device()` 的对应 API。

**返回值:** int – 当前 CUDA 设备索引

**注意:** 
- 如果当前没有设置设备或 CUDA 不可用，可能会抛出异常
- 建议在使用前先检查 `is_available()`

**例子:**
```cpp
#include "origin.h"

#ifdef WITH_CUDA
    if (origin::cuda::is_available()) {
        // 设置设备
        origin::cuda::set_device(0);
        
        // 获取当前设备
        int current = origin::cuda::current_device();
        std::cout << "Current device: " << current << std::endl;
    }
#endif
```

### origin::cuda::set_device

```cpp
void origin::cuda::set_device(int device_id)
```

设置当前 CUDA 设备。这是 PyTorch 中 `torch.cuda.set_device(device)` 的对应 API。

**参数:**
- `device_id` (int) – 要设置的设备 ID，必须小于 `device_count()`

**注意:** 
- 如果设备 ID 无效，会抛出异常
- 建议在使用前先检查 `device_count()` 确保设备 ID 有效

**例子:**
```cpp
#include "origin.h"

#ifdef WITH_CUDA
    if (origin::cuda::is_available()) {
        int count = origin::cuda::device_count();
        
        if (count > 0) {
            // 设置使用第一个设备
            origin::cuda::set_device(0);
            
            // 创建 CUDA 张量（会使用当前设置的设备）
            auto t = Tensor::ones({2, 2}, 
                                  TensorOptions().device(DeviceType::kCUDA));
        }
    }
#endif
```

### origin::cuda::device_info

```cpp
void origin::cuda::device_info()
```

打印所有可用 CUDA 设备的详细信息，包括设备名称、计算能力、内存大小、多处理器数量等。

**例子:**
```cpp
#include "origin.h"

#ifdef WITH_CUDA
    if (origin::cuda::is_available()) {
        // 打印所有设备信息
        origin::cuda::device_info();
        // 输出示例:
        // CUDA devices available: 2
        // Device 0: NVIDIA GeForce RTX 3090
        //   Compute capability: 8.6
        //   Memory: 24564 MB
        //   Multiprocessors: 82
        //   Max threads per block: 1024
        // Device 1: NVIDIA GeForce RTX 3080
        //   ...
    }
#endif
```



---

## 当前实现限制

本文档列出了 OriginDL 当前版本的一些限制和与 PyTorch 的差异。这些限制可能会在未来的版本中得到改进。

### 数学函数限制

#### sin 和 cos 函数

**限制**: `sin()` 和 `cos()` 函数在 Origin 后端尚未实现。

**当前状态**: 调用这些函数会抛出 `RuntimeError` 异常。

**影响范围**: 仅影响 Origin 后端，如果使用 Torch 后端则不受影响。

**例子:**
```cpp
auto t = Tensor::ones({2, 2});
auto s = sin(t);  // 抛出异常: "sin function not implemented yet"
auto c = cos(t);  // 抛出异常: "cos function not implemented yet"
```

#### log 函数

**限制**: `log()` 函数在 Origin 后端仅在 CPU 上实现，CUDA 张量会回退到 CPU 计算。

**当前状态**: 即使 CUDA 有实现，`OriginMat::log()` 也只调用 CPU 版本。

**影响范围**: CUDA 张量的 `log()` 操作会先复制到 CPU，计算后再复制回 CUDA，影响性能。

### 矩阵乘法限制

**限制**: `mat_mul()` 仅支持以下两种形状组合：
- `{m, k} x {k, n}` → `{m, n}` (2D x 2D)
- `{batch, m, k} x {k, n}` → `{batch, m, n}` (3D x 2D)

**不支持的形式**:
- `{batch, m, k} x {batch, k, n}` (3D x 3D)
- `{m, k} x {batch, k, n}` (2D x 3D，广播)
- 其他维度组合

**与 PyTorch 的差异**: PyTorch 的 `torch.matmul` 支持更广泛的形状组合，包括批量矩阵乘法和广播。

### sum_to 限制

**限制**: `sum_to()` 不支持广播。

**行为**: 当目标形状的元素数量大于源张量时，会抛出异常（与 libtorch 一致，但 libtorch 会静默返回原始张量）。

**与 PyTorch 的差异**: PyTorch 没有直接的 `sum_to` 函数，但可以通过 `sum` + `expand` 实现广播功能。

**例子:**
```cpp
auto x = Tensor({5.0}, Shape{1});
auto result = sum_to(x, Shape{3});  // 抛出异常: 不支持广播
```

### CUDA 支持限制

#### 复杂广播

**限制**: CUDA 后端不支持复杂广播操作。

**影响范围**: 对于需要复杂广播的算术运算（如 `add`, `subtract`, `multiply` 等），如果形状不匹配且不是简单的标量广播，会抛出异常。

**当前支持**: 仅支持相同形状或标量广播（其中一个张量元素数量为1）。

**例子:**
```cpp
// 在CUDA上
auto a = Tensor::ones({2, 3}, device(DeviceType::kCUDA));
auto b = Tensor::ones({3, 2}, device(DeviceType::kCUDA));
auto c = a + b;  // 可能抛出异常: "Complex broadcasting not yet implemented"
```



### 自动求导限制

#### requires_grad

**限制**: 当前不支持 `requires_grad=false`。

**当前状态**: 所有张量默认 `requires_grad=true`，无法禁用梯度计算。

**影响**: 
- 所有张量都会参与梯度计算，即使不需要梯度
- 内存占用可能较大
- 无法优化不需要梯度的计算图

**例子:**
```cpp
// 当前行为：requires_grad 参数被忽略，总是为 true
auto x = Tensor::ones({2, 2}, requires_grad(false));  // 实际上仍然是 true
```

#### 梯度类型

**限制**: `grad()` 方法总是返回 `float32` 类型的梯度，而不是与输入张量相同的类型。

**当前状态**: 即使输入张量是 `float64` 或其他类型，梯度也是 `float32`。

**影响**: 可能影响高精度计算的准确性。

**例子:**
```cpp
auto x = Tensor::ones({2, 2}, dtype(DataType::kFloat64));
auto y = x * x;
y.backward();
auto g = x.grad();  // g.dtype() 是 kFloat32，而不是 kFloat64
```

### 形状操作限制

#### 视图转置

**限制**: Origin 后端不支持视图转置（view transpose）。

**当前状态**: `transpose()` 总是创建新的张量，而不是返回视图。

**影响**: 转置操作会复制数据，内存开销较大。

**与 PyTorch 的差异**: PyTorch 的转置在某些情况下可以返回视图，不复制数据。

### 反向传播中的类型提升

**限制**: 某些算子的反向传播中尚未实现类型提升逻辑。

**影响范围**: `add`, `sub`, `mul`, `div`, `mat_mul` 等算子的反向传播。

**当前状态**: 如果前向传播中进行了类型提升，反向传播可能无法正确处理。

**注意**: 前向传播中的类型提升是正常工作的。

### 其他限制

#### 布尔类型运算

**限制**: 布尔类型张量的某些运算有特殊行为或限制。

- **除法**: 布尔类型不支持除法操作，会抛出异常
- **加法**: 布尔类型加法使用逻辑 OR 作为替代
- **减法**: 布尔类型减法使用逻辑异或 (XOR) 作为替代
- **乘法**: 布尔类型乘法使用逻辑 AND

**例子:**
```cpp
auto a = Tensor({true, false}, Shape{2}, DataType::kBool);
auto b = Tensor({true, true}, Shape{2}, DataType::kBool);
auto c = a + b;   // 使用逻辑 OR: [true, true]
auto d = a * b;   // 使用逻辑 AND: [true, false]
auto e = a / b;   // 抛出异常: "Division is not supported for boolean tensors"
```