#include "base/dlException.h"
#include "dlTensorImpl.h"
#include "dlOperator.h"

namespace dl
{

TensorImpl& TensorImpl::operator=(const TensorImpl &other) {
    if (this != &other) {
        data_ = other.data_;
        grad_ = other.grad_;
        creator_ = other.creator_;
        generation_ = other.generation_;
    }
    return *this;
}

TensorImpl& TensorImpl::operator=(TensorImpl &&other) noexcept {
    if (this != &other) {
        data_ = std::move(other.data_);
        grad_ = std::move(other.grad_);
        creator_ = std::move(other.creator_);
        generation_ = other.generation_;
    }
    return *this;
}

void TensorImpl::set_creator(const FunctionPtr &func) {
    creator_ = func;
    generation_ = creator_->generation_ + 1;
}

void TensorImpl::backward() {
    if (grad_.elements() == 0) {
        double grad_val = 1.0;
        auto dims = this->data_.dims();
        grad_ = af::constant(grad_val, dims);
    }

    auto funcs = std::list<FunctionPtr>();
    auto func_set = std::set<FunctionPtr>();

    auto add_func = [&funcs, &func_set](const FunctionPtr &f) {
        if (func_set.find(f) == func_set.end()) {
            funcs.push_back(f);
            func_set.insert(f);
            funcs.sort(
                [](const FunctionPtr &lhs, const FunctionPtr &rhs) { return lhs->generation_ < rhs->generation_; });
        }
    };

    add_func(this->creator_);

    while (!funcs.empty()) {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = std::vector<Tensor>();
        // 检查 outputs_ 是否为空
        if (f->outputs_.empty()) {
            DL_ERROR_THROW("outputs_ is empty");
        }
        for (const auto &o : f->outputs_) {
            // 检查 shared_ptr 是否为空
            if (!o) {
                DL_ERROR_THROW("outputs_ contains null shared_ptr");
            }
            // 关键修复：传递的应该是输出张量的梯度，而不是数据值
            // 创建一个包含梯度数据的 Tensor 对象
            gys.push_back(Tensor(o->grad()));
        }
        auto gxs = f->backward(gys);

        if (gxs.size() != f->inputs_.size()) {
            DL_ERROR_THROW("backward error!, gxs size " + std::to_string(gxs.size()) + ", inputs size " +
                           std::to_string(f->inputs_.size()));
        }

        for (size_t i = 0; i < gxs.size(); i++) {
            auto x = f->inputs_[i];
            auto gx = gxs[i];

            // 梯度累积逻辑：如果梯度为空，直接赋值；否则累加
            // 注意：这里需要检查梯度是否真正为空
            if (x.grad().elements() == 0) {
                x.grad() = gx.data();
            } else {
                x.grad() = x.grad() + gx.data();
            }

            if (x.get_impl()->creator_) {
                add_func(x.get_impl()->creator_);
            }
        }
    }
}

void TensorImpl::clear_grad() {
    // 将梯度设置为与数据相同形状的零数组
    grad_ = af::constant(0.0, data_.dims(), data_.type());
}

TensorImpl TensorImpl::reshape(const af::dim4 &shape) const {
    auto result = TensorImpl(af::moddims(data_, shape));
    return result;
}

TensorImpl TensorImpl::transpose() const {
    auto result = TensorImpl(af::transpose(data_));
    return result;
}

TensorImpl TensorImpl::operator+(const TensorImpl &other) const {
    return TensorImpl(data_ + other.data_);
}

TensorImpl TensorImpl::operator+(data_t scalar) const {
    return TensorImpl(data_ + scalar);
}

TensorImpl TensorImpl::operator-(const TensorImpl &other) const {
    return TensorImpl(data_ - other.data_);
}

TensorImpl TensorImpl::operator*(const TensorImpl &other) const {
    return TensorImpl(data_ * other.data_);
}

TensorImpl TensorImpl::operator/(const TensorImpl &other) const {
    return TensorImpl(data_ / other.data_);
}

TensorImpl TensorImpl::operator-() const {
    return TensorImpl(-data_);
}

void TensorImpl::print(const std::string &desc) const {
    af::print(desc.c_str(), data_);
}

}  // namespace dl
