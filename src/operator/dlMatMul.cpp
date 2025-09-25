#include "dlOperator.h"

namespace dl
{

std::vector<Tensor> MatMul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2) {
        throw std::runtime_error("MatMul requires exactly 2 inputs");
    }
    
    auto x = xs[0].data();
    auto w = xs[1].data();
    auto y = af::matmul(x, w);
    return std::vector<Tensor>{Tensor(y)};
}

std::vector<Tensor> MatMul::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1) {
        throw std::runtime_error("MatMul backward requires exactly 1 gradient");
    }

    auto x = this->inputs_[0].data();
    auto w = this->inputs_[1].data();
    auto gy = gys[0].data();
    
    auto gx = Tensor(af::matmul(gy, w.T()));
    auto gw = Tensor(af::matmul(x.T(), gy));
    
    return std::vector<Tensor>{gx, gw};
}

Tensor matmul(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<MatMul>();
    return (*op)(xs)[0];
}

Tensor matmul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul({lhs, rhs});
}

Tensor mat_mul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul(lhs, rhs);
}

}  // namespace dl