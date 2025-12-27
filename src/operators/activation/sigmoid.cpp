#include "origin/core/operator.h"
#include "origin/utils/exception.h"
#include "origin/mat/mat.h"

namespace origin
{

std::vector<Tensor> Sigmoid::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Sigmoid operator requires exactly 1 input, but got {}", xs.size());
    }

    // Sigmoid: y = 1 / (1 + exp(-x))
    auto &x = xs[0];
    
    // 计算 -x
    auto neg_x_result = -mat(x);
    auto neg_x = convert_mat_to_tensor(std::move(neg_x_result));
    
    // 计算 exp(-x)
    auto exp_neg_x = exp(neg_x);
    
    // 创建全1张量，形状与exp_neg_x相同
    auto ones = Tensor::ones(exp_neg_x.shape(), dtype(exp_neg_x.dtype()).device(exp_neg_x.device()));
    
    // 计算 1 + exp(-x)
    auto one_plus_exp = ones + exp_neg_x;
    
    // 计算 1 / (1 + exp(-x))
    auto result = ones / one_plus_exp;
    
    std::vector<Tensor> outputs;
    outputs.push_back(result);
    return outputs;
}

std::vector<Tensor> Sigmoid::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Sigmoid backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // Sigmoid 的梯度：gx = gy * sigmoid(x) * (1 - sigmoid(x))
    auto &x = this->inputs_[0];
    auto &gy = gys[0];
    
    // 计算 sigmoid(x)
    auto sigmoid_x_result = -mat(x);
    auto neg_x = convert_mat_to_tensor(std::move(sigmoid_x_result));
    auto exp_neg_x = exp(neg_x);
    auto ones = Tensor::ones(exp_neg_x.shape(), dtype(exp_neg_x.dtype()).device(exp_neg_x.device()));
    auto one_plus_exp = ones + exp_neg_x;
    auto sigmoid_x = ones / one_plus_exp;
    
    // 计算 1 - sigmoid(x)
    auto one_minus_sigmoid = ones - sigmoid_x;
    
    // 计算 gx = gy * sigmoid(x) * (1 - sigmoid(x))
    auto temp = gy * sigmoid_x;
    auto gx = temp * one_minus_sigmoid;
    
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor sigmoid(const Tensor &x)
{
    auto op = std::make_shared<Sigmoid>();
    return (*op)(x);
}

}  // namespace origin

