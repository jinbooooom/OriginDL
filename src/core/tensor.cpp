#include "origin/core/tensor.h"
#include <stdexcept>
#include "origin/core/tensor_options.h"
#include "origin/mat/backend.h"
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

namespace origin
{

// === 构造函数和析构函数实现 ===

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

// 从Mat创建Tensor的构造函数实现
Tensor::Tensor(std::unique_ptr<Mat> mat) : impl_(std::make_shared<TensorImpl>(std::move(mat))) {}

// === 工厂方法实现 ===

Tensor Tensor::zeros(const Shape &shape, const TensorOptions &options)
{
    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            std::vector<float> data(shape.elements(), 0.0f);
            return Tensor(data, shape, options);
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> data(shape.elements(), 0);
            return Tensor(data, shape, options);
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> data(shape.elements(), 0);
            return Tensor(data, shape, options);
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for zeros operation", dtype_to_string(options.dtype()));
    }
}

Tensor Tensor::ones(const Shape &shape, const TensorOptions &options)
{
    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            std::vector<float> data(shape.elements(), 1.0f);
            return Tensor(data, shape, options);
        }
        case DataType::kInt32:
        {
            std::vector<int32_t> data(shape.elements(), 1);
            return Tensor(data, shape, options);
        }
        case DataType::kInt8:
        {
            std::vector<int8_t> data(shape.elements(), 1);
            return Tensor(data, shape, options);
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for ones operation", dtype_to_string(options.dtype()));
    }
}

Tensor Tensor::randn(const Shape &shape, const TensorOptions &options)
{
    auto impl = TensorImpl::randn(shape, options);
    return Tensor(std::make_shared<TensorImpl>(std::move(impl)));
}

Tensor Tensor::full(const Shape &shape, double value, const TensorOptions &options)
{
    switch (options.dtype())
    {
        case DataType::kFloat32:
            return Tensor(static_cast<float>(value), shape, options);
        case DataType::kDouble:
            return Tensor(value, shape, options);
        case DataType::kInt32:
            return Tensor(static_cast<int32_t>(value), shape, options);
        case DataType::kInt8:
            return Tensor(static_cast<int8_t>(value), shape, options);
        default:
            THROW_INVALID_ARG("Unsupported data type {} for full operation", dtype_to_string(options.dtype()));
    }
}

Tensor Tensor::from_blob(void *data, const Shape &shape, const TensorOptions &options)
{
    Tensor result;
    result.create_tensor_from_raw_data(data, shape, options.dtype());
    // 如果设备不是CPU，需要移动到指定设备
    if (options.device().type() != DeviceType::kCPU)
    {
        result = result.to(options);
    }
    return result;
}

// === 形状和维度实现 ===

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

// === 类型查询和转换实现 ===

DataType Tensor::dtype() const
{
    return impl_->data_->dtype();
}

Tensor Tensor::to(DataType target_type) const
{
    auto converted_mat = impl_->data_->to(target_type);
    return Tensor(std::make_unique<TensorImpl>(std::move(converted_mat)));
}

Tensor Tensor::to(const TensorOptions &options) const
{
    auto converted_impl = impl_->to(options);
    return Tensor(std::make_shared<TensorImpl>(std::move(converted_impl)));
}

// === 梯度相关实现 ===

Tensor Tensor::grad() const
{
    if (!impl_->grad_)
    {
        return Tensor::zeros(shape(), origin::dtype(DataType::kFloat32));  // TODO，创建与input同类型的gtad
    }
    // 通过TensorImpl创建，避免直接类型转换
    return Tensor(impl_->grad_->clone());
}

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

// === 张量操作实现 ===

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
// 注意：标量操作使用全局操作符重载，避免与成员操作符冲突

// === 调试实现 ===

void Tensor::print(const std::string &desc) const
{
    impl_->print(desc);
}

template <typename T>
std::vector<T> Tensor::to_vector() const
{
    return impl_->to_vector<T>();
}

int Tensor::backend_type() const
{
    return impl_->backend_type();
}

// === 私有方法实现 ===

void Tensor::create_tensor_from_raw_data(const void *data, const Shape &shape, DataType dtype)
{
    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                              shape.to_string());
        }
    }

    // 最底层方法：直接通过TensorImpl创建，避免任何中间拷贝
    impl_ = std::make_unique<TensorImpl>(data, shape, dtype);
}

template <typename T>
void Tensor::create_tensor_from_scalar_with_dtype(T scalar, const Shape &shape, DataType dtype)
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type (int, float, double, etc.)");
    static_assert(!std::is_pointer_v<T>, "T cannot be a pointer type");

    // 验证形状是否有效
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                              shape.to_string());
        }
    }

    switch (dtype)
    {
        case DataType::kFloat32:
        {
            float val = static_cast<float>(scalar);
            impl_     = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        case DataType::kDouble:
        {
            double val = static_cast<double>(scalar);
            impl_      = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        case DataType::kInt32:
        {
            int32_t val = static_cast<int32_t>(scalar);
            impl_       = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        case DataType::kInt8:
        {
            int8_t val = static_cast<int8_t>(scalar);
            impl_      = std::make_unique<TensorImpl>(val, shape);
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported data type {} for tensor creation", dtype_to_string(dtype));
    }
}

template <typename T>
void Tensor::create_tensor_from_data_with_dtype(const T *data, size_t count, const Shape &shape, DataType dtype)
{
    // 验证数据大小
    if (count != shape.elements())
    {
        THROW_INVALID_ARG("Data count {} does not match shape elements {}", count, shape.elements());
    }

    // 验证形状是否有效
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                              shape.to_string());
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
            THROW_INVALID_ARG("Unsupported data type {} for tensor creation", dtype_to_string(dtype));
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
// 成员操作符模板实例化已移除，使用全局操作符重载

// 内部辅助方法
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

// 模板实例化 - create_tensor_from_scalar_with_dtype
template void Tensor::create_tensor_from_scalar_with_dtype<float>(float scalar, const Shape &shape, DataType dtype);
template void Tensor::create_tensor_from_scalar_with_dtype<double>(double scalar, const Shape &shape, DataType dtype);
template void Tensor::create_tensor_from_scalar_with_dtype<int32_t>(int32_t scalar, const Shape &shape, DataType dtype);
template void Tensor::create_tensor_from_scalar_with_dtype<int8_t>(int8_t scalar, const Shape &shape, DataType dtype);
template void Tensor::create_tensor_from_scalar_with_dtype<size_t>(size_t scalar, const Shape &shape, DataType dtype);

}  // namespace origin