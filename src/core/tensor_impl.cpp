#include "origin/core/tensor_impl.h"
#include <list>
#include <set>
#include <stdexcept>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/backend.h"
#include "origin/utils/exception.h"

namespace origin
{

// 从数据创建TensorImpl的构造函数实现
TensorImpl::TensorImpl(const std::vector<data_t> &data, const Shape &shape)
    : data_(std::make_unique<Mat_t>(data, shape)), grad_(nullptr), creator_(nullptr), generation_(0)
{
    // 验证数据是否为空
    if (data.empty())
    {
        throw std::invalid_argument("Tensor data cannot be empty. Data vector is empty.");
    }

    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("Tensor shape cannot have zero dimensions. Dimension " + std::to_string(i) +
                                        " is zero in shape " + shape.to_string());
        }
    }

    // 验证数据大小与形状是否匹配
    size_t expected_elements = shape.elements();
    if (data.size() != expected_elements)
    {
        throw std::invalid_argument("Data size (" + std::to_string(data.size()) + ") does not match shape elements (" +
                                    std::to_string(expected_elements) + ")");
    }
}

TensorImpl::TensorImpl(data_t scalar, const Shape &shape)
    : data_(std::make_unique<Mat_t>(scalar, shape)), grad_(nullptr), creator_(nullptr), generation_(0)
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
}

// 静态工厂方法实现
TensorImpl TensorImpl::randn(const Shape &shape)
{
    // 通过后端Mat接口创建随机数矩阵
    auto mat = Mat_t::randn(shape);
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
    if (!grad_)
    {
        grad_ = std::make_unique<Mat_t>(1.0, data_->shape());
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

TensorImpl TensorImpl::operator+(data_t scalar) const
{
    auto result = *data_ + scalar;
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

data_t TensorImpl::item() const
{
    if (elements() != 1)
    {
        throw std::runtime_error("item() can only be called on scalar tensors");
    }
    return data_->to_vector()[0];
}

std::vector<data_t> TensorImpl::to_vector() const
{
    return data_->to_vector();
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

}  // namespace origin