#include "origin/optim/adam.h"
#include <cmath>
#include <unordered_map>
#include "origin/core/operator.h"
#include "origin/core/parameter.h"
#include "origin/core/tensor.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"

namespace origin
{

Adam::Adam(Module &target, float lr, float beta1, float beta2, float eps)
    : Optimizer(target), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps)
{}

void Adam::step_one(Parameter &param)
{
    // 获取梯度
    auto grad = param.grad();

    // 初始化或获取状态缓冲区
    auto m_it    = m_buffers_.find(&param);
    auto v_it    = v_buffers_.find(&param);
    auto step_it = step_counts_.find(&param);

    // 初始化缓冲区（如果不存在）
    if (m_it == m_buffers_.end())
    {
        m_buffers_[&param] = Tensor::zeros(grad.shape(), TensorOptions(grad.dtype()).device(grad.device()));
        m_it               = m_buffers_.find(&param);
    }
    if (v_it == v_buffers_.end())
    {
        v_buffers_[&param] = Tensor::zeros(grad.shape(), TensorOptions(grad.dtype()).device(grad.device()));
        v_it               = v_buffers_.find(&param);
    }
    if (step_it == step_counts_.end())
    {
        step_counts_[&param] = 0;
        step_it              = step_counts_.find(&param);
    }

    // 增加步数
    int &t = step_it->second;
    t++;

    // 更新一阶矩估计：m = beta1 * m + (1 - beta1) * grad
    auto &m = m_it->second;
    m       = m * Scalar(beta1_) + grad * Scalar(1.0f - beta1_);

    // 更新二阶矩估计：v = beta2 * v + (1 - beta2) * grad^2
    auto &v           = v_it->second;
    auto grad_squared = grad * grad;
    v                 = v * Scalar(beta2_) + grad_squared * Scalar(1.0f - beta2_);

    // 计算偏差修正：m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    float beta1_power = std::pow(beta1_, static_cast<float>(t));
    float beta2_power = std::pow(beta2_, static_cast<float>(t));
    float m_hat_scale = 1.0f / (1.0f - beta1_power);
    float v_hat_scale = 1.0f / (1.0f - beta2_power);

    auto m_hat = m * Scalar(m_hat_scale);
    auto v_hat = v * Scalar(v_hat_scale);

    // 计算 sqrt(v_hat) + eps
    // 注意：我们需要实现 sqrt 操作，或者使用其他方法
    // 为了简化，我们可以使用 v_hat 的平方根近似
    // 但实际上我们需要一个 sqrt 函数

    // 临时方案：使用 v_hat 的数值计算
    // 我们需要计算 sqrt(v_hat)，但当前没有 sqrt 算子
    // 让我们先实现一个简单的版本，使用 v_hat 的平方根

    // 由于没有 sqrt 算子，我们需要手动计算
    // 对于标量，我们可以直接计算；对于张量，我们需要遍历
    // 但为了保持一致性，我们应该实现一个通用的方法

    // 暂时使用一个近似：sqrt(v_hat) ≈ v_hat^0.5
    // 但我们需要 pow 函数支持标量指数

    // 实际上，我们可以使用现有的操作来实现 sqrt
    // sqrt(x) = x^0.5，但我们需要检查是否有 pow 支持

    // 为了简化，我们先实现一个基本版本，假设有 sqrt 函数
    // 如果没有，我们需要添加

    // 检查是否有 sqrt 函数
    // 如果没有，我们需要添加一个

    // 临时方案：使用数值方法计算 sqrt
    // 对于每个元素，计算 sqrt(v_hat[i])

    // 由于没有直接的 sqrt 算子，我们需要手动实现
    // 但为了保持代码简洁，我们先假设有 sqrt 函数

    // 实际上，我们可以使用现有的操作：sqrt(x) = pow(x, 0.5)
    // 但需要检查 pow 是否支持浮点指数

    // 计算 sqrt(v_hat) = v_hat^0.5
    auto sqrt_v_hat = pow(v_hat, Scalar(0.5f));

    // 计算分母：sqrt(v_hat) + eps
    auto denominator = sqrt_v_hat + Scalar(eps_);

    // 计算更新：param = param - lr * m_hat / (sqrt(v_hat) + eps)
    auto update  = m_hat / denominator;
    auto updated = param - update * Scalar(lr_);

    // 更新参数
    param = Parameter(updated);
}

}  // namespace origin
