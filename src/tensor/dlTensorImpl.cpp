#include "../include/dlTensorImpl.h"
#include <list>
#include <set>
#include <stdexcept>
#include "../include/base/dlException.h"
#include "../include/dlOperator.h"
#include "../include/dlTensor.h"
#include "../include/mat/dlArrayFireMat.h"

namespace dl
{

// 赋值运算符实现
TensorImpl &TensorImpl::operator=(const TensorImpl &other)
{
    if (this != &other)
    {
        data_    = other.data_ ? std::unique_ptr<Mat_t>(static_cast<Mat_t *>(other.data_->clone().release())) : nullptr;
        grad_    = other.grad_ ? std::unique_ptr<Mat_t>(static_cast<Mat_t *>(other.grad_->clone().release())) : nullptr;
        creator_ = other.creator_;
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
                x.impl_->grad_   = std::move(new_grad);
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
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(new_mat.release())));
}

TensorImpl TensorImpl::transpose() const
{
    auto new_mat = data_->transpose();
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(new_mat.release())));
}

// 运算符重载实现
TensorImpl TensorImpl::operator+(const TensorImpl &other) const
{
    auto result = *data_ + *other.data_;
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(result.release())));
}

TensorImpl TensorImpl::operator+(data_t scalar) const
{
    auto result = *data_ + scalar;
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(result.release())));
}

TensorImpl TensorImpl::operator-(const TensorImpl &other) const
{
    auto result = *data_ - *other.data_;
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(result.release())));
}

TensorImpl TensorImpl::operator*(const TensorImpl &other) const
{
    auto result = *data_ * *other.data_;
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(result.release())));
}

TensorImpl TensorImpl::operator/(const TensorImpl &other) const
{
    auto result = *data_ / *other.data_;
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(result.release())));
}

TensorImpl TensorImpl::operator-() const
{
    auto result = -*data_;
    return TensorImpl(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(result.release())));
}

// 调试方法实现
void TensorImpl::print(const std::string &desc) const
{
    if (data_)
    {
        data_->print(desc);
    }
}

}  // namespace dl