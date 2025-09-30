#include "origin/core/tensor.h"
#include <stdexcept>
#include "origin/mat/array_fire_mat.h"
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

// 公共构造函数实现
Tensor::Tensor(const std::vector<data_t> &data, const Shape &shape)
{
    // 通过TensorImpl创建，Tensor只能调用TensorImpl的方法
    impl_ = std::make_shared<TensorImpl>(data, shape);
}

Tensor::Tensor(std::initializer_list<data_t> data, const Shape &shape)
{
    std::vector<data_t> data_vec(data);
    // 通过TensorImpl创建，Tensor只能调用TensorImpl的方法
    impl_ = std::make_shared<TensorImpl>(data_vec, shape);
}

Tensor::Tensor(data_t scalar, const Shape &shape)
{
    // 通过TensorImpl创建，Tensor只能调用TensorImpl的方法
    impl_ = std::make_shared<TensorImpl>(scalar, shape);
}

// 工厂函数实现
Tensor Tensor::zeros(const Shape &shape)
{
    std::vector<data_t> data(shape.elements(), 0.0);
    return Tensor(data, shape);
}

Tensor Tensor::ones(const Shape &shape)
{
    std::vector<data_t> data(shape.elements(), 1.0);
    return Tensor(data, shape);
}

Tensor Tensor::randn(const Shape &shape)
{
    // 通过TensorImpl的randn方法创建，符合架构设计
    auto impl = TensorImpl::randn(shape);
    return Tensor(std::make_shared<TensorImpl>(std::move(impl)));
}

Tensor Tensor::constant(data_t value, const Shape &shape)
{
    return Tensor(value, shape);
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

data_t Tensor::item() const
{
    return impl_->item();
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

// 数据转换
std::vector<data_t> Tensor::to_vector() const
{
    return impl_->to_vector();
}

}  // namespace origin