#ifndef __ORIGIN_DL_ADAM_H__
#define __ORIGIN_DL_ADAM_H__

#include "../core/operator.h"
#include "../core/tensor.h"
#include "optimizer.h"

namespace origin
{

/**
 * @brief Adam优化器
 * @details 实现自适应矩估计（Adam）优化算法
 * 
 * Adam算法：
 * - m = beta1 * m + (1 - beta1) * grad
 * - v = beta2 * v + (1 - beta2) * grad^2
 * - m_hat = m / (1 - beta1^t)
 * - v_hat = v / (1 - beta2^t)
 * - param = param - lr * m_hat / (sqrt(v_hat) + eps)
 */
class Adam : public Optimizer
{
private:
    float lr_;      // 学习率
    float beta1_;   // 一阶矩估计的衰减率，默认为0.9
    float beta2_;   // 二阶矩估计的衰减率，默认为0.999
    float eps_;     // 数值稳定性常数，默认为1e-8

    // 状态字典
    std::unordered_map<Parameter *, Tensor> m_buffers_;  // 一阶矩估计
    std::unordered_map<Parameter *, Tensor> v_buffers_;  // 二阶矩估计
    std::unordered_map<Parameter *, int> step_counts_;   // 每个参数的步数计数

public:
    /**
     * @brief 构造函数
     * @param target 目标模块
     * @param lr 学习率
     * @param beta1 一阶矩估计的衰减率，默认为0.9
     * @param beta2 二阶矩估计的衰减率，默认为0.999
     * @param eps 数值稳定性常数，默认为1e-8
     */
    Adam(Module &target, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);

protected:
    /**
     * @brief 更新单个参数
     * @param param 参数引用
     */
    void step_one(Parameter &param) override;
};

}  // namespace origin

#endif  // __ORIGIN_DL_ADAM_H__

