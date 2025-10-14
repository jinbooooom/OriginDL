#include "origin/core/tensor_impl.h"
#include <list>
#include <set>
#include <stdexcept>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/core/tensor_options.h"
#include "origin/mat/backend.h"
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

namespace origin
{

// 从void*数据构造的实现
TensorImpl::TensorImpl(const void *data, const Shape &shape, DataType dtype)
    : grad_(nullptr), creator_(nullptr), generation_(0)
{
    create_impl_from_data(data, shape, dtype);
}

// 从void*数据构造的实现（支持TensorOptions）
TensorImpl::TensorImpl(const void *data, const Shape &shape, const TensorOptions &options)
    : grad_(nullptr), creator_(nullptr), generation_(0)
{
    create_impl_from_data(data, shape, options.dtype());
    // 如果设备不是CPU，需要移动到指定设备
    if (options.device().type() != DeviceType::kCPU)
    {
        data_ = data_->to_device(options.device());
    }
}

// 静态工厂方法实现
TensorImpl TensorImpl::randn(const Shape &shape)
{
    // 通过后端Mat接口创建随机数矩阵
    auto mat = Mat_t::randn(shape);
    return TensorImpl(std::move(mat));
}

TensorImpl TensorImpl::randn(const Shape &shape, const TensorOptions &options)
{
    // 通过后端Mat接口创建随机数矩阵
    auto mat = Mat_t::randn(shape, options);
    return TensorImpl(std::move(mat));
}

// 赋值运算符实现
TensorImpl &TensorImpl::operator=(const TensorImpl &other)
{
    if (this != &other)
    {
        data_       = other.data_ ? other.data_->clone() : nullptr;
        grad_       = other.grad_ ? other.grad_->clone() : nullptr;
        creator_    = other.creator_;
        generation_ = other.generation_;
    }
    return *this;
}

TensorImpl &TensorImpl::operator=(TensorImpl &&other) noexcept
{
    if (this != &other)
    {
        data_       = std::move(other.data_);
        grad_       = std::move(other.grad_);
        creator_    = std::move(other.creator_);
        generation_ = other.generation_;
    }
    return *this;
}

void TensorImpl::set_creator(const FunctionPtr &func)
{
    creator_    = func;
    generation_ = creator_->generation_ + 1;
}

void TensorImpl::backward()
{
    // 如果梯度为空，初始化为全1（输出张量的梯度）
    // 梯度类型应与数据类型一致
    if (!grad_)
    {
        auto data_type = data_->dtype();
        switch (data_type)
        {
            case DataType::kFloat32:
                grad_ = std::make_unique<Mat_t>(1.0f, data_->shape());
                break;
            case DataType::kDouble:
                grad_ = std::make_unique<Mat_t>(1.0, data_->shape());
                break;
            case DataType::kInt32:
                grad_ = std::make_unique<Mat_t>(1, data_->shape());
                break;
            case DataType::kInt8:
                grad_ = std::make_unique<Mat_t>(static_cast<int8_t>(1), data_->shape());
                break;
            default:
                throw std::invalid_argument("Unsupported data type for gradient initialization");
        }
    }

    auto funcs    = std::list<FunctionPtr>();
    auto func_set = std::set<FunctionPtr>();

    auto add_func = [&funcs, &func_set](const FunctionPtr &f) {
        if (f && func_set.find(f) == func_set.end())
        {
            funcs.push_back(f);
            func_set.insert(f);
            funcs.sort(
                [](const FunctionPtr &lhs, const FunctionPtr &rhs) { return lhs->generation_ < rhs->generation_; });
        }
    };

    add_func(this->creator_);

    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = std::vector<Tensor>();
        // 检查 outputs_ 是否为空
        if (f->outputs_.empty())
        {
            DL_ERROR_THROW("outputs_ is empty");
        }
        for (const auto &o : f->outputs_)
        {
            // 检查 shared_ptr 是否为空
            if (!o)
            {
                DL_ERROR_THROW("outputs_ contains null shared_ptr");
            }
            // 获取输出张量的梯度
            gys.push_back(Tensor(o->grad()));
        }
        auto gxs = f->backward(gys);

        if (gxs.size() != f->inputs_.size())
        {
            DL_ERROR_THROW("backward error!, gxs size " + std::to_string(gxs.size()) + ", inputs size " +
                           std::to_string(f->inputs_.size()));
        }

        for (size_t i = 0; i < gxs.size(); i++)
        {
            auto x  = f->inputs_[i];
            auto gx = gxs[i];

            // 梯度累积逻辑：如果梯度为空，直接赋值；否则累加
            if (!x.impl_->grad_)
            {
                // 梯度为空，直接赋值
                x.impl_->grad_ = gx.impl_->data_->clone();
            }
            else
            {
                // 梯度不为空，累加
                auto current_grad = x.impl_->grad_->clone();
                auto new_grad     = *current_grad + *gx.impl_->data_;
                x.impl_->grad_    = std::move(new_grad);
            }

            if (x.impl_->creator_)
            {
                add_func(x.impl_->creator_);
            }
        }
    }
}

void TensorImpl::clear_grad()
{
    grad_ = nullptr;
}

// 张量操作实现
TensorImpl TensorImpl::reshape(const Shape &shape) const
{
    auto new_mat = data_->reshape(shape);
    return TensorImpl(std::move(new_mat));
}

TensorImpl TensorImpl::transpose() const
{
    auto new_mat = data_->transpose();
    return TensorImpl(std::move(new_mat));
}

// 运算符重载实现
TensorImpl TensorImpl::operator+(const TensorImpl &other) const
{
    auto result = *data_ + *other.data_;
    return TensorImpl(std::move(result));
}

template <typename T>
TensorImpl TensorImpl::operator+(T scalar) const
{
    auto result = data_->add_scalar<T>(scalar);
    return TensorImpl(std::move(result));
}

TensorImpl TensorImpl::operator-(const TensorImpl &other) const
{
    auto result = *data_ - *other.data_;
    return TensorImpl(std::move(result));
}

TensorImpl TensorImpl::operator*(const TensorImpl &other) const
{
    auto result = *data_ * *other.data_;
    return TensorImpl(std::move(result));
}

TensorImpl TensorImpl::operator/(const TensorImpl &other) const
{
    auto result = *data_ / *other.data_;
    return TensorImpl(std::move(result));
}

TensorImpl TensorImpl::operator-() const
{
    auto result = -*data_;
    return TensorImpl(std::move(result));
}

// 访问器方法实现
Shape TensorImpl::shape() const
{
    return data_->shape();
}

size_t TensorImpl::ndim() const
{
    return data_->shape().size();
}

size_t TensorImpl::elements() const
{
    return data_->elements();
}

template <typename T>
T TensorImpl::item() const
{
    if (elements() != 1)
    {
        throw std::runtime_error("item() can only be called on scalar tensors");
    }
    return data_->to_vector<T>()[0];
}

template <typename T>
std::vector<T> TensorImpl::to_vector() const
{
    return data_->to_vector<T>();
}

// === 泛型数据访问方法实现 ===

template <typename T>
T *TensorImpl::data_ptr()
{
    return data_->data_ptr<T>();
}

// === 泛型标量操作实现 ===

template <typename T>
TensorImpl TensorImpl::operator-(T scalar) const
{
    auto result = data_->operator-(scalar);
    return TensorImpl(std::move(result));
}

template <typename T>
TensorImpl TensorImpl::operator*(T scalar) const
{
    auto result = data_->mul_scalar<T>(scalar);
    return TensorImpl(std::move(result));
}

template <typename T>
TensorImpl TensorImpl::operator/(T scalar) const
{
    auto result = data_->operator/(scalar);
    return TensorImpl(std::move(result));
}

int TensorImpl::backend_type() const
{
    return data_->backend_type();
}

// 调试方法实现
void TensorImpl::print(const std::string &desc) const
{
    if (data_)
    {
        data_->print(desc);
    }
}

// 类型转换实现
TensorImpl TensorImpl::to(const TensorOptions &options) const
{
    auto converted_mat = data_->to(options.dtype());
    if (options.device().type() != DeviceType::kCPU)
    {
        converted_mat = converted_mat->to_device(options.device());
    }
    return TensorImpl(std::move(converted_mat));
}

// 私有辅助方法实现
void TensorImpl::create_impl_from_data(const void *data, const Shape &shape, DataType dtype)
{
    size_t count = shape.elements();
    switch (dtype)
    {
        case DataType::kFloat32:
            create_impl_impl<float>(static_cast<const float *>(data), count, shape);
            break;
        case DataType::kDouble:
            create_impl_impl<double>(static_cast<const double *>(data), count, shape);
            break;
        case DataType::kInt32:
            create_impl_impl<int32_t>(static_cast<const int32_t *>(data), count, shape);
            break;
        case DataType::kInt8:
            create_impl_impl<int8_t>(static_cast<const int8_t *>(data), count, shape);
            break;
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

void TensorImpl::create_impl_from_scalar(double data, const Shape &shape, DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            create_impl_impl<float>(static_cast<float>(data), shape);
            break;
        case DataType::kDouble:
            create_impl_impl<double>(data, shape);
            break;
        case DataType::kInt32:
            create_impl_impl<int32_t>(static_cast<int32_t>(data), shape);
            break;
        case DataType::kInt8:
            create_impl_impl<int8_t>(static_cast<int8_t>(data), shape);
            break;
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

// 模板函数实现
template <typename T>
void TensorImpl::create_impl_impl(const T *data, size_t count, const Shape &shape)
{
    std::vector<T> vec_data(data, data + count);
    data_       = std::make_unique<Mat_t>(vec_data, shape);
    grad_       = nullptr;
    creator_    = nullptr;
    generation_ = 0;
}

template <typename T>
void TensorImpl::create_impl_impl(T scalar, const Shape &shape)
{
    data_       = std::make_unique<Mat_t>(scalar, shape);
    grad_       = nullptr;
    creator_    = nullptr;
    generation_ = 0;
}

// 显式实例化常用类型
template void TensorImpl::create_impl_impl<float>(const float *data, size_t count, const Shape &shape);
template void TensorImpl::create_impl_impl<double>(const double *data, size_t count, const Shape &shape);
template void TensorImpl::create_impl_impl<int32_t>(const int32_t *data, size_t count, const Shape &shape);
template void TensorImpl::create_impl_impl<int8_t>(const int8_t *data, size_t count, const Shape &shape);
template void TensorImpl::create_impl_impl<float>(float scalar, const Shape &shape);
template void TensorImpl::create_impl_impl<double>(double scalar, const Shape &shape);
template void TensorImpl::create_impl_impl<int32_t>(int32_t scalar, const Shape &shape);
template void TensorImpl::create_impl_impl<int8_t>(int8_t scalar, const Shape &shape);

// === 泛型方法实例化 ===
// 数据访问方法
template float TensorImpl::item<float>() const;
template double TensorImpl::item<double>() const;
template int32_t TensorImpl::item<int32_t>() const;
template int8_t TensorImpl::item<int8_t>() const;

template float *TensorImpl::data_ptr<float>();
template double *TensorImpl::data_ptr<double>();
template int32_t *TensorImpl::data_ptr<int32_t>();
template int8_t *TensorImpl::data_ptr<int8_t>();

template std::vector<float> TensorImpl::to_vector<float>() const;
template std::vector<double> TensorImpl::to_vector<double>() const;
template std::vector<int32_t> TensorImpl::to_vector<int32_t>() const;
template std::vector<int8_t> TensorImpl::to_vector<int8_t>() const;

// 泛型标量操作
template TensorImpl TensorImpl::operator+<float>(float scalar) const;
template TensorImpl TensorImpl::operator+<double>(double scalar) const;
template TensorImpl TensorImpl::operator+<int32_t>(int32_t scalar) const;
template TensorImpl TensorImpl::operator+<int8_t>(int8_t scalar) const;

template TensorImpl TensorImpl::operator-<float>(float scalar) const;
template TensorImpl TensorImpl::operator-<double>(double scalar) const;
template TensorImpl TensorImpl::operator-<int32_t>(int32_t scalar) const;
template TensorImpl TensorImpl::operator-<int8_t>(int8_t scalar) const;

template TensorImpl TensorImpl::operator*<float>(float scalar) const;
template TensorImpl TensorImpl::operator*<double>(double scalar) const;
template TensorImpl TensorImpl::operator*<int32_t>(int32_t scalar) const;
template TensorImpl TensorImpl::operator*<int8_t>(int8_t scalar) const;

template TensorImpl TensorImpl::operator/<float>(float scalar) const;
template TensorImpl TensorImpl::operator/<double>(double scalar) const;
template TensorImpl TensorImpl::operator/<int32_t>(int32_t scalar) const;
template TensorImpl TensorImpl::operator/<int8_t>(int8_t scalar) const;

// 额外的模板实例化（只添加新的类型）
template TensorImpl TensorImpl::operator+<unsigned long>(unsigned long scalar) const;
template unsigned long TensorImpl::item<unsigned long>() const;
template unsigned long *TensorImpl::data_ptr<unsigned long>();
template std::vector<unsigned long> TensorImpl::to_vector<unsigned long>() const;

}  // namespace origin