#include "dlTensor.h"
#include <stdexcept>
#include "../include/mat/dlArrayFireMat.h"
#include "base/dlException.h"

namespace dl
{

// 公共构造函数实现
Tensor::Tensor(const std::vector<data_t> &data, const Shape &shape)
{
    auto mat = std::make_unique<Mat_t>(data, shape);
    impl_    = std::make_shared<TensorImpl>(std::move(mat));
}

Tensor::Tensor(std::initializer_list<data_t> data, const Shape &shape)
{
    std::vector<data_t> data_vec(data);
    auto mat = std::make_unique<Mat_t>(data_vec, shape);
    impl_    = std::make_shared<TensorImpl>(std::move(mat));
}

Tensor::Tensor(data_t scalar, const Shape &shape)
{
    auto mat = std::make_unique<Mat_t>(scalar, shape);
    impl_    = std::make_shared<TensorImpl>(std::move(mat));
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

// TODO：Tensor 中直接调用 ArrayFireMat 的静态方法不是一个好的设计。
// 可以改成先创建 impl，通过 impl 的方法去创建 Mat，然后用 impl 创建 Tensor
Tensor Tensor::randn(const Shape &shape)
{
    af::array rand_array = af::randn(ArrayFireMat::convert_shape_to_af_dim4(shape));
    return Tensor(std::make_unique<Mat_t>(std::move(rand_array)));
}

Tensor Tensor::constant(data_t value, const Shape &shape)
{
    return Tensor(std::make_unique<Mat_t>(value, shape));
}

Tensor Tensor::from_data(const std::vector<data_t> &data, const Shape &shape)
{
    return Tensor(data, shape);
}

// 公共访问器实现
Shape Tensor::shape() const
{
    return impl_->data_->shape();
}

size_t Tensor::ndim() const
{
    return impl_->data_->shape().size();
}

size_t Tensor::elements() const
{
    return impl_->data_->elements();
}

data_t Tensor::item() const
{
    if (elements() != 1)
    {
        throw std::runtime_error("item() can only be called on scalar tensors");
    }
    return impl_->data_->to_vector()[0];
}

Tensor Tensor::grad() const
{
    if (!impl_->grad_)
    {
        return Tensor::zeros(shape());
    }
    return Tensor(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(impl_->grad_->clone().release())));
}

// 张量操作实现
Tensor Tensor::reshape(const Shape &shape) const
{
    auto new_mat = impl_->data_->reshape(shape);
    return Tensor(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(new_mat.release())));
}

Tensor Tensor::transpose() const
{
    auto new_mat = impl_->data_->transpose();
    return Tensor(std::unique_ptr<Mat_t>(static_cast<Mat_t *>(new_mat.release())));
}

// 数据转换
std::vector<data_t> Tensor::to_vector() const
{
    return impl_->data_->to_vector();
}

}  // namespace dl