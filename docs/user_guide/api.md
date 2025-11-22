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

获取张量的梯度。如果梯度未初始化，将返回一个全零张量。

**返回值:** Tensor – 梯度张量

**例子:**
```cpp
auto x = Tensor::ones({2, 2});
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

**返回值:** Tensor – 转换后的张量

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat32));

// 转换数据类型
auto t_float64 = t.to(DataType::kFloat64);

// 转换设备
auto t_cuda = t.to(Device(DeviceType::kCUDA));

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

**返回值:** T – 标量值

**例子:**
```cpp
auto t = Tensor::full({1}, 3.14f);
float value = t.item<float>();  // 3.14
// t.print() 输出:
// 3.14
//  OriginMat(shape={}, dtype=float32, device=cpu)
```

### data_ptr

```cpp
template <typename T>
T *data_ptr()
```

获取张量数据的原始指针。

**返回值:** T* – 数据指针

**例子:**
```cpp
auto t = Tensor::ones({2, 2}, dtype(DataType::kFloat32));
float* ptr = t.data_ptr<float>();
// 现在可以通过ptr访问数据
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

**例子:**
```cpp
auto a = Tensor::ones({2, 2});
auto b = a + 2.0;  // 标量加法
auto c = 3.0 * a;  // 标量乘法
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
auto e = a ^ 2.5;        // 使用操作符形式
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

矩阵乘法。

**例子:**
```cpp
auto a = Tensor::ones({2, 3});
auto b = Tensor::ones({3, 4});
auto c = mat_mul(a, b);
// 结果: 2x4的矩阵
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