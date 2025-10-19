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

// 两个核心工厂方法实现
TensorImpl TensorImpl::from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
{
    // 直接调用OriginMat工厂方法
    auto mat = OriginMat::from_scalar(scalar, shape, options);
    return TensorImpl(std::move(mat));
}

TensorImpl TensorImpl::from_memory(const void *data, const Shape &shape, const TensorOptions &options)
{
    // 直接调用OriginMat工厂方法
    auto mat = OriginMat::from_memory(data, shape, options);
    return TensorImpl(std::move(mat));
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
            {
                Scalar scalar_val(1.0f);
                TensorOptions options(DataType::kFloat32);
                grad_ = OriginMat::from_scalar(scalar_val, data_->shape(), options);
                break;
            }
            case DataType::kDouble:
            {
                Scalar scalar_val(1.0);
                TensorOptions options(DataType::kDouble);
                grad_ = OriginMat::from_scalar(scalar_val, data_->shape(), options);
                break;
            }
            case DataType::kInt32:
            {
                Scalar scalar_val(1);
                TensorOptions options(DataType::kInt32);
                grad_ = OriginMat::from_scalar(scalar_val, data_->shape(), options);
                break;
            }
            case DataType::kInt8:
            {
                Scalar scalar_val(static_cast<int8_t>(1));
                TensorOptions options(DataType::kInt8);
                grad_ = OriginMat::from_scalar(scalar_val, data_->shape(), options);
                break;
            }
            default:
                THROW_INVALID_ARG("Unsupported data type {} for gradient initialization",
                                  dtype_to_string(data_->dtype()));
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
            THROW_RUNTIME_ERROR("outputs_ is empty");
        }
        for (const auto &o : f->outputs_)
        {
            // 检查 shared_ptr 是否为空
            if (!o)
            {
                THROW_RUNTIME_ERROR("outputs_ contains null shared_ptr");
            }
            // 获取输出张量的梯度
            gys.push_back(Tensor(o->grad()));
        }
        auto gxs = f->backward(gys);

        if (gxs.size() != f->inputs_.size())
        {
            THROW_RUNTIME_ERROR("backward error!, gxs size {} inputs size {}", gxs.size(), f->inputs_.size());
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
        THROW_RUNTIME_ERROR("item() can only be called on scalar tensors, but tensor has {} elements", elements());
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

// 移除所有私有辅助方法，直接实现核心逻辑

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