#include "origin/core/tensor.h"
#include <stdexcept>
#include "origin/mat/backend.h"
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

namespace origin
{

// 内部构造函数实现
Tensor::Tensor(TensorImplPtr impl) : impl_(impl) {}

// 拷贝和移动构造函数实现
Tensor::Tensor(const Tensor &other) : impl_(other.impl_) {}

Tensor::Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}

// 赋值运算符实现
Tensor &Tensor::operator=(const Tensor &other)
{
    impl_ = other.impl_;
    return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept
{
    impl_ = std::move(other.impl_);
    return *this;
}

// 方法委托实现
void Tensor::set_creator(const FunctionPtr &func)
{
    impl_->set_creator(func);
}

void Tensor::backward()
{
    impl_->backward();
}

void Tensor::clear_grad()
{
    impl_->clear_grad();
}

// 调试方法实现
void Tensor::print(const std::string &desc) const
{
    impl_->print(desc);
}

// 从Mat创建Tensor的构造函数实现
Tensor::Tensor(std::unique_ptr<Mat> mat) : impl_(std::make_shared<TensorImpl>(std::move(mat))) {}

// 工厂函数实现
Tensor Tensor::zeros(const Shape &shape, DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
        {
            std::vector<float> data(shape.elements(), 0.0f);
            return Tensor(data, shape);
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> data(shape.elements(), 0);
            return Tensor(data, shape);
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> data(shape.elements(), 0);
            return Tensor(data, shape);
        }
        default:
            throw std::invalid_argument("Unsupported data type for zeros");
    }
}

Tensor Tensor::ones(const Shape &shape, DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
        {
            std::vector<float> data(shape.elements(), 1.0f);
            return Tensor(data, shape);
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> data(shape.elements(), 1);
            return Tensor(data, shape);
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> data(shape.elements(), 1);
            return Tensor(data, shape);
        }
        default:
            throw std::invalid_argument("Unsupported data type for ones");
    }
}

Tensor Tensor::randn(const Shape &shape, DataType dtype)
{
    // 目前randn只支持float32，其他类型需要先创建float32再转换
    auto impl   = TensorImpl::randn(shape);
    auto tensor = Tensor(std::make_shared<TensorImpl>(std::move(impl)));

    if (dtype == DataType::kFloat32)
    {
        return tensor;
    }
    else
    {
        return tensor.to(dtype);
    }
}

// === 显式类型构造函数实现 ===
Tensor Tensor::from_blob(void *data, const Shape &shape, DataType dtype)
{
    Tensor result;
    result.create_tensor_from_raw_data(data, shape, dtype);
    return result;
}

// === 工厂方法实现 ===
Tensor Tensor::full(const Shape &shape, double value, DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return Tensor(static_cast<float>(value), shape);
        case DataType::kDouble:
            return Tensor(value, shape);
        case DataType::kInt32:
            return Tensor(static_cast<int32_t>(value), shape);
        case DataType::kInt8:
            return Tensor(static_cast<int8_t>(value), shape);
        default:
            throw std::invalid_argument("Unsupported data type for full");
    }
}

// 公共访问器实现
Shape Tensor::shape() const
{
    return impl_->shape();
}

size_t Tensor::ndim() const
{
    return impl_->ndim();
}

size_t Tensor::elements() const
{
    return impl_->elements();
}

// === 数据访问：类型安全实现 ===
template <typename T>
T Tensor::item() const
{
    return impl_->item<T>();
}

template <typename T>
T *Tensor::data_ptr()
{
    return impl_->data_ptr<T>();
}

template <typename T>
std::vector<T> Tensor::to_vector() const
{
    return impl_->to_vector<T>();
}

Tensor Tensor::grad() const
{
    if (!impl_->grad_)
    {
        return Tensor::zeros(shape());
    }
    // 通过TensorImpl创建，避免直接类型转换
    return Tensor(impl_->grad_->clone());
}

// 张量操作实现
Tensor Tensor::reshape(const Shape &shape) const
{
    // 通过TensorImpl的reshape方法，避免直接操作Mat
    auto new_impl = impl_->reshape(shape);
    return Tensor(std::make_shared<TensorImpl>(std::move(new_impl)));
}

Tensor Tensor::transpose() const
{
    // 通过TensorImpl的transpose方法，避免直接操作Mat
    auto new_impl = impl_->transpose();
    return Tensor(std::make_shared<TensorImpl>(std::move(new_impl)));
}

// === 泛型标量操作实现 ===
template <typename T>
Tensor Tensor::operator+(T scalar) const
{
    auto result = impl_->operator+(scalar);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
}

template <typename T>
Tensor Tensor::operator-(T scalar) const
{
    auto result = impl_->operator-(scalar);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
}

template <typename T>
Tensor Tensor::operator*(T scalar) const
{
    auto result = impl_->operator*(scalar);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
}

template <typename T>
Tensor Tensor::operator/(T scalar) const
{
    auto result = impl_->operator/(scalar);
    return Tensor(std::make_shared<TensorImpl>(std::move(result)));
}

// 后端信息
int Tensor::backend_type() const
{
    return impl_->backend_type();
}

// 类型转换和查询
Tensor Tensor::to(DataType target_type) const
{
    auto converted_mat = impl_->data_->to(target_type);
    return Tensor(std::make_unique<TensorImpl>(std::move(converted_mat)));
}

DataType Tensor::dtype() const
{
    return impl_->data_->dtype();
}

// 指定数据类型的构造函数实现
Tensor::Tensor(const void *data, const Shape &shape, DataType dtype)
{
    create_tensor_from_raw_data(data, shape, dtype);
}

void Tensor::create_tensor_from_raw_data(const void *data, const Shape &shape, DataType dtype)
{
    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("Tensor shape cannot have zero dimensions. Dimension " + std::to_string(i) +
                                        " is zero in shape " + shape.to_string());
        }
    }

    // 最底层方法：直接通过TensorImpl创建，避免任何中间拷贝
    impl_ = std::make_unique<TensorImpl>(data, shape, dtype);
}

// === 用于自动类型推断的方法实现 ===
template <typename T>
void Tensor::create_tensor_from_scalar(T data, const Shape &shape)
{
    auto inferred_type = get_data_type<T>();
    create_tensor_from_scalar_with_dtype(&data, shape, inferred_type);
}

template <typename T>
void Tensor::create_tensor_from_data(const T *data, size_t count, const Shape &shape)
{
    auto inferred_type = get_data_type<T>();
    create_tensor_from_data_with_dtype(data, count, shape, inferred_type);
}

// === 用于显式类型指定的方法实现 ===
void Tensor::create_tensor_from_scalar_with_dtype(const void *data, const Shape &shape, DataType dtype)
{
    // 验证形状是否有效
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("Tensor shape cannot have zero dimensions. Dimension " + std::to_string(i) +
                                        " is zero in shape " + shape.to_string());
        }
    }

    switch (dtype)
    {
        case DataType::kFloat32:
        {
            float val = *static_cast<const float *>(data);
            impl_     = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        case DataType::kDouble:
        {
            double val = *static_cast<const double *>(data);
            impl_      = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        case DataType::kInt32:
        {
            int32_t val = *static_cast<const int32_t *>(data);
            impl_       = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        case DataType::kInt8:
        {
            int8_t val = *static_cast<const int8_t *>(data);
            impl_      = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

template <typename T>
void Tensor::create_tensor_from_data_with_dtype(const T *data, size_t count, const Shape &shape, DataType dtype)
{
    // 验证数据大小
    if (count != shape.elements())
    {
        throw std::invalid_argument("Data count does not match shape elements");
    }

    // 验证形状是否有效
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("Tensor shape cannot have zero dimensions. Dimension " + std::to_string(i) +
                                        " is zero in shape " + shape.to_string());
        }
    }

    // 根据目标类型进行转换
    switch (dtype)
    {
        case DataType::kFloat32:
        {
            std::vector<float> converted_data(count);
            for (size_t i = 0; i < count; ++i)
            {
                converted_data[i] = static_cast<float>(data[i]);
            }
            impl_ = std::make_unique<TensorImpl>(converted_data, shape);
            break;
        }
        case DataType::kDouble:
        {
            std::vector<double> converted_data(count);
            for (size_t i = 0; i < count; ++i)
            {
                converted_data[i] = static_cast<double>(data[i]);
            }
            impl_ = std::make_unique<TensorImpl>(converted_data, shape);
            break;
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> converted_data(count);
            for (size_t i = 0; i < count; ++i)
            {
                converted_data[i] = static_cast<int32_t>(data[i]);
            }
            impl_ = std::make_unique<TensorImpl>(converted_data, shape);
            break;
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> converted_data(count);
            for (size_t i = 0; i < count; ++i)
            {
                converted_data[i] = static_cast<int8_t>(data[i]);
            }
            impl_ = std::make_unique<TensorImpl>(converted_data, shape);
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

// === 模板实例化 ===
// 数据访问方法
template float Tensor::item<float>() const;
template double Tensor::item<double>() const;
template int32_t Tensor::item<int32_t>() const;
template int8_t Tensor::item<int8_t>() const;
template unsigned long Tensor::item<unsigned long>() const;

template float *Tensor::data_ptr<float>();
template double *Tensor::data_ptr<double>();
template int32_t *Tensor::data_ptr<int32_t>();
template int8_t *Tensor::data_ptr<int8_t>();
template unsigned long *Tensor::data_ptr<unsigned long>();

template std::vector<float> Tensor::to_vector<float>() const;
template std::vector<double> Tensor::to_vector<double>() const;
template std::vector<int32_t> Tensor::to_vector<int32_t>() const;
template std::vector<int8_t> Tensor::to_vector<int8_t>() const;
template std::vector<unsigned long> Tensor::to_vector<unsigned long>() const;

// 泛型标量操作
template Tensor Tensor::operator+<float>(float scalar) const;
template Tensor Tensor::operator+<double>(double scalar) const;
template Tensor Tensor::operator+<int32_t>(int32_t scalar) const;
template Tensor Tensor::operator+<int8_t>(int8_t scalar) const;

template Tensor Tensor::operator-<float>(float scalar) const;
template Tensor Tensor::operator-<double>(double scalar) const;
template Tensor Tensor::operator-<int32_t>(int32_t scalar) const;
template Tensor Tensor::operator-<int8_t>(int8_t scalar) const;

template Tensor Tensor::operator*<float>(float scalar) const;
template Tensor Tensor::operator*<double>(double scalar) const;
template Tensor Tensor::operator*<int32_t>(int32_t scalar) const;
template Tensor Tensor::operator*<int8_t>(int8_t scalar) const;

template Tensor Tensor::operator/<float>(float scalar) const;
template Tensor Tensor::operator/<double>(double scalar) const;
template Tensor Tensor::operator/<int32_t>(int32_t scalar) const;
template Tensor Tensor::operator/<int8_t>(int8_t scalar) const;

// 内部辅助方法
template void Tensor::create_tensor_from_scalar<float>(float data, const Shape &shape);
template void Tensor::create_tensor_from_scalar<double>(double data, const Shape &shape);
template void Tensor::create_tensor_from_scalar<int32_t>(int32_t data, const Shape &shape);
template void Tensor::create_tensor_from_scalar<int8_t>(int8_t data, const Shape &shape);
template void Tensor::create_tensor_from_scalar<unsigned long>(unsigned long data, const Shape &shape);

template void Tensor::create_tensor_from_data<float>(const float *data, size_t count, const Shape &shape);
template void Tensor::create_tensor_from_data<double>(const double *data, size_t count, const Shape &shape);
template void Tensor::create_tensor_from_data<int32_t>(const int32_t *data, size_t count, const Shape &shape);
template void Tensor::create_tensor_from_data<int8_t>(const int8_t *data, size_t count, const Shape &shape);
template void Tensor::create_tensor_from_data<unsigned long>(const unsigned long *data,
                                                             size_t count,
                                                             const Shape &shape);

template void Tensor::create_tensor_from_data_with_dtype<float>(const float *data,
                                                                size_t count,
                                                                const Shape &shape,
                                                                DataType dtype);
template void Tensor::create_tensor_from_data_with_dtype<double>(const double *data,
                                                                 size_t count,
                                                                 const Shape &shape,
                                                                 DataType dtype);
template void Tensor::create_tensor_from_data_with_dtype<int32_t>(const int32_t *data,
                                                                  size_t count,
                                                                  const Shape &shape,
                                                                  DataType dtype);
template void Tensor::create_tensor_from_data_with_dtype<int8_t>(const int8_t *data,
                                                                 size_t count,
                                                                 const Shape &shape,
                                                                 DataType dtype);

}  // namespace origin