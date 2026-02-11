#ifndef __ORIGIN_DL_HOOKS_H__
#define __ORIGIN_DL_HOOKS_H__

#include <functional>
#include <vector>
#include "../core/operator.h"  // 需要包含operator.h以使用operator*和broadcast_to
#include "../core/parameter.h"
#include "../mat/scalar.h"

namespace origin
{

/**
 * @brief 权重衰减Hook
 *
 * 在优化器更新参数之前，对梯度添加权重衰减项
 * 公式：grad = grad + rate * param
 *
 * 这等价于在损失函数中添加 L2 正则化项：loss = loss + 0.5 * rate * ||param||^2
 */
class WeightDecay
{
private:
    float rate_;  // 权重衰减率

public:
    /**
     * @brief 构造函数
     * @param rate 权重衰减率，默认为 1e-4
     */
    explicit WeightDecay(float rate = 1e-4f) : rate_(rate) {}

    /**
     * @brief 获取Hook函数
     * @return Hook函数，可以在优化器中注册
     */
    std::function<void(std::vector<Parameter *> &)> hook()
    {
        float rate = rate_;
        return [rate](std::vector<Parameter *> &params) {
            for (auto *param : params)
            {
                if (param == nullptr)
                {
                    continue;
                }

                // 计算权重衰减项：rate * param
                // operator*需要const Tensor&，所以直接使用const引用
                const Tensor &param_tensor = static_cast<const Tensor &>(*param);
                // 使用Scalar乘法，会自动广播到param的形状
                auto weight_decay_term = param_tensor * Scalar(rate);

                // 确保weight_decay_term的形状与梯度匹配
                // 获取当前梯度形状（如果梯度为空，使用参数形状）
                auto grad_shape = param->grad().shape();
                if (weight_decay_term.shape() != grad_shape)
                {
                    // 如果形状不匹配，需要广播（通常不会发生，因为param和grad形状相同）
                    weight_decay_term = functional::broadcast_to(weight_decay_term, grad_shape);
                }

                // 累加权重衰减到梯度：grad = grad + rate * param
                param->accumulate_grad(weight_decay_term);
            }
        };
    }
};

}  // namespace origin

#endif  // __ORIGIN_DL_HOOKS_H__
