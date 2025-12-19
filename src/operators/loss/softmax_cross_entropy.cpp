#include "origin/core/operator.h"
#include "origin/utils/exception.h"
#include "origin/mat/mat.h"
#include <cmath>
#include <algorithm>

namespace origin
{

std::vector<Tensor> SoftmaxCrossEntropy::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("SoftmaxCrossEntropy operator requires exactly 2 inputs (x and target), but got {}", xs.size());
    }

    auto &x = xs[0];
    auto &target = xs[1];
    
    auto x_shape = x.shape();
    auto target_shape = target.shape();
    
    // 验证输入形状
    if (x_shape.size() != 2)
    {
        THROW_INVALID_ARG("SoftmaxCrossEntropy expects x to be 2D (N, C), but got shape {}", x_shape.to_string());
    }
    if (target_shape.size() != 1)
    {
        THROW_INVALID_ARG("SoftmaxCrossEntropy expects target to be 1D (N,), but got shape {}", target_shape.to_string());
    }
    if (x_shape[0] != target_shape[0])
    {
        THROW_INVALID_ARG("SoftmaxCrossEntropy: batch size mismatch. x has {} samples, target has {} samples", 
                         x_shape[0], target_shape[0]);
    }
    
    size_t N = x_shape[0];  // batch size
    size_t C = x_shape[1];  // number of classes
    
    // 1. 计算 softmax: p = softmax(x)
    auto p = softmax(x, -1);  // 沿最后一个维度计算 softmax
    
    // 2. 计算交叉熵: loss = -mean(log(p[target]))
    // 对于每个样本 i，选择 p[i][target[i]]，然后取 log
    auto p_data = p.to_vector<float>();
    auto target_data = target.to_vector<int32_t>();
    
    float total_loss = 0.0f;
    for (size_t i = 0; i < N; ++i)
    {
        int32_t t = target_data[i];
        if (t < 0 || t >= static_cast<int32_t>(C))
        {
            THROW_INVALID_ARG("SoftmaxCrossEntropy: target index {} out of range [0, {})", t, C);
        }
        
        // p[i][t] = p_data[i * C + t]
        float prob = p_data[i * C + t];
        if (prob <= 0.0f)
        {
            // 避免 log(0)，使用一个很小的值
            prob = 1e-8f;
        }
        total_loss += std::log(prob);
    }
    
    // loss = -mean(log(p[target]))
    float loss_value = -total_loss / static_cast<float>(N);
    
    // 创建标量损失张量
    auto loss = Tensor({loss_value}, Shape{}, dtype(DataType::kFloat32).device(x.device()));
    
    std::vector<Tensor> outputs;
    outputs.push_back(loss);
    return outputs;
}

std::vector<Tensor> SoftmaxCrossEntropy::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("SoftmaxCrossEntropy backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // 梯度计算：gx = (softmax(x) - one_hot(target)) / N
    auto &x = this->inputs_[0];
    auto &target = this->inputs_[1];
    auto &gy = gys[0];
    
    auto x_shape = x.shape();
    size_t N = x_shape[0];  // batch size
    size_t C = x_shape[1];  // number of classes
    
    // 1. 计算 softmax(x)
    auto p = softmax(x, -1);
    
    // 2. 创建 one_hot(target) 编码
    auto target_data = target.to_vector<int32_t>();
    std::vector<float> one_hot_data(N * C, 0.0f);
    for (size_t i = 0; i < N; ++i)
    {
        int32_t t = target_data[i];
        if (t >= 0 && t < static_cast<int32_t>(C))
        {
            one_hot_data[i * C + t] = 1.0f;
        }
    }
    auto one_hot = Tensor(one_hot_data, x_shape, dtype(DataType::kFloat32).device(x.device()));
    
    // 3. 计算 gx = (softmax(x) - one_hot(target)) / N
    auto diff = p - one_hot;
    
    // 4. 应用梯度缩放（gy 通常是 1.0，但为了通用性，我们乘以它）
    auto gy_value = gy.item<float>();
    auto gx = diff * (gy_value / static_cast<float>(N));
    
    // 5. target 不需要梯度（它是标签）
    auto gtarget = Tensor::zeros(target.shape(), dtype(DataType::kFloat32).device(target.device()));
    
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(gtarget);
    return outputs;
}

Tensor softmax_cross_entropy(const Tensor &x, const Tensor &target)
{
    auto op = std::make_shared<SoftmaxCrossEntropy>();
    std::vector<Tensor> inputs = {x, target};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace origin

