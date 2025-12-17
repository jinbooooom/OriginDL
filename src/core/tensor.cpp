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
    return Tensor(0, shape, options);
}

Tensor Tensor::ones(const Shape &shape, const TensorOptions &options)
{
    return Tensor(1, shape, options);
}

Tensor Tensor::randn(const Shape &shape, const TensorOptions &options)
{
    auto impl = TensorImpl::randn(shape, options);
    return Tensor(std::make_shared<TensorImpl>(std::move(impl)));
}

Tensor Tensor::full(const Shape &shape, const Scalar &value, const TensorOptions &options)
{
    return Tensor(value, shape, options);
}

Tensor Tensor::from_blob(void *data, const Shape &shape, const TensorOptions &options)
{
    Tensor result;
    result.from_memory(data, options.dtype(), shape, options);

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

// === 张量属性方法实现 ===

size_t Tensor::element_size() const
{
    return origin::element_size(dtype()); // 返回单个元素占用的字节数
}

size_t Tensor::numel() const
{
    return elements();  // numel()和elements()功能相同
}

size_t Tensor::nbytes() const
{
    return element_size() * numel();
}

// === 数据访问：类型安全实现 ===

template <typename T>
T Tensor::item() const
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    return impl_->item<T>();
}

template <typename T>
T *Tensor::data_ptr()
{
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    return impl_->data_ptr<T>();
}

// === 类型查询和转换实现 ===

DataType Tensor::dtype() const
{
    return impl_->data_->dtype();
}

Device Tensor::device() const
{
    return impl_->data_->device();
}

Tensor Tensor::to(DataType target_type) const
{
    auto converted_mat = impl_->data_->to(target_type);
    return Tensor(std::make_unique<TensorImpl>(std::move(converted_mat)));
}

Tensor Tensor::to(Device device) const
{
    auto converted_mat = impl_->data_->to_device(device);
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
        return Tensor::zeros(shape(), origin::dtype(DataType::kFloat32).device(device()));  // TODO，创建与input同类型的gtad
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
    ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    return impl_->to_vector<T>();
}

int Tensor::backend_type() const
{
    return impl_->backend_type();
}

// === 私有方法实现 ===

void Tensor::from_memory(const void *data, DataType user_dtype, const Shape &shape, const TensorOptions &options)
{
    // 验证形状是否有效
    // 0维张量（标量张量）是合法的，但其他维度不能为0
    if (!shape.is_scalar())
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
            {
                THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                                  shape.to_string());
            }
        }
    }

    // 直接调用TensorImpl工厂方法
    impl_ = std::make_unique<TensorImpl>(TensorImpl::from_memory(data, user_dtype, shape, options));
}

void Tensor::from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
{
    // 验证形状是否有效
    // 0维张量（标量张量）是合法的，但其他维度不能为0
    if (!shape.is_scalar())
    {
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
            {
                THROW_INVALID_ARG("Tensor shape cannot have zero dimensions. Dimension {} is zero in shape {}", i,
                                  shape.to_string());
            }
        }
    }

    // 直接调用TensorImpl工厂方法并设置impl_
    impl_ = std::make_unique<TensorImpl>(TensorImpl::from_scalar(scalar, shape, options));
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

}  // namespace origin