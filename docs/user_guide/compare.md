# OriginDL 与 PyTorch 对比

本文档对比了 OriginDL 和 PyTorch 的 API 使用方式，帮助用户从 PyTorch 迁移到 OriginDL。

## 张量创建

### ones

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 创建全1张量 | `torch.ones(2, 3, dtype="float32", device='cuda:0')` | `Tensor::ones({2, 3}, dtype("float32").device("cuda:0"))` | OriginDL使用Shape对象和链式配置，语法高度相似 |

### zeros

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 创建全0张量 | `torch.zeros(2, 3, dtype="float32", device='cuda:0')` | `Tensor::zeros({2, 3}, dtype("float32").device("cuda:0"))` | 语法高度相似 |

### randn

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 创建随机张量 | `torch.randn(2, 3, dtype="float32", device='cuda:0')` | `Tensor::randn({2, 3}, dtype("float32").device("cuda:0"))` | 语法高度相似 |

### full

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 创建填充张量 | `torch.full((2, 3), 5.0, dtype="float32", device='cuda:0')` | `Tensor::full({2, 3}, 5.0, dtype("float32").device("cuda:0"))` | OriginDL使用Shape对象，PyTorch使用元组 |

### from_numpy

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 从numpy创建 | `torch.from_numpy(arr)` | `Tensor::from_blob(arr.data, {2, 3}, dtype("float32").device("cuda:0"))` | OriginDL需要显式指定形状和选项 |

## 张量属性

### shape

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 获取形状 | `tensor.shape` | `tensor.shape()` | OriginDL使用函数调用 |

### dtype

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 获取数据类型 | `tensor.dtype` | `tensor.dtype()` | OriginDL使用函数调用 |

### device

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 获取设备 | `tensor.device` | `tensor.device()` | OriginDL使用函数调用 |

## 张量操作

### reshape

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 重塑形状 | `tensor.reshape(3, 2)` | `tensor.reshape({3, 2})` | OriginDL使用Shape对象 |

### transpose

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 转置张量 | `tensor.transpose()` | `tensor.transpose()` | 语法完全一致 |

## 数学运算

### 加法

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 张量加法 | `a + b` | `a + b` | 语法完全一致 |

### 乘法

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 张量乘法 | `a * b` | `a * b` | 语法完全一致 |

### 矩阵乘法

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 矩阵乘法 | `torch.matmul(a, b)` | `mat_mul(a, b)` | OriginDL使用下划线命名 |

### 减法

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 张量减法 | `a - b` | `a - b` | 语法完全一致 |

### 除法

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 张量除法 | `a / b` | `a / b` | 语法完全一致 |

### 取负

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 张量取负 | `-a` | `-a` | 语法完全一致 |

## 数学函数

### square

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 平方运算 | `torch.square(a)` | `square(a)` | OriginDL使用函数形式 |

### pow

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 幂运算 | `torch.pow(a, 2)` | `pow(a, 2)` | 语法完全一致 |

### exp

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 指数函数 | `torch.exp(a)` | `exp(a)` | 语法完全一致 |

## 归约操作

### sum

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 求和 | `torch.sum(a)` | `sum(a)` | 语法完全一致 |
| 按轴求和 | `torch.sum(a, dim=0)` | `sum(a, 0)` | OriginDL使用位置参数 |

## 形状操作

### broadcast_to

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 广播 | `torch.broadcast_to(a, (3, 2))` | `broadcast_to(a, {3, 2})` | OriginDL使用Shape对象 |

### sum_to

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 求和到形状 | `torch.sum_to(a, (1, 2))` | `sum_to(a, {1, 2})` | OriginDL使用Shape对象 |

## 类型转换

### to

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 转换数据类型 | `tensor.to(dtype="float64")` | `tensor.to(DataType::kFloat64)` | OriginDL使用枚举 |
| 转换设备 | `tensor.to(device="cuda")` | `tensor.to(DeviceType::kCUDA)` | OriginDL使用枚举 |
| 链式转换 | `tensor.to(dtype="float64", device="cuda")` | `tensor.to(dtype("float64").device("cuda"))` | OriginDL使用链式调用 |

## 数据访问

### item

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 获取标量值 | `tensor.item()` | `tensor.item<float>()` | OriginDL需要指定类型 |

### data_ptr

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 获取数据指针 | `tensor.data_ptr()` | `tensor.data_ptr<float>()` | OriginDL需要指定类型 |

## 调试工具

### print

| 功能 | PyTorch 示例代码 | OriginDL 示例代码 | 备注 |
|------|------------------|-------------------|------|
| 打印张量 | `print(tensor)` | `tensor.print()` | OriginDL使用成员函数 |
| 带描述打印 | `print("desc:", tensor)` | `tensor.print("desc")` | OriginDL使用成员函数 |




