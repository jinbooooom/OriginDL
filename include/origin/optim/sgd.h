#ifndef __ORIGIN_DL_SGD_H__
#define __ORIGIN_DL_SGD_H__

#include "../core/operator.h"
#include "../core/tensor.h"
#include "optimizer.h"

namespace origin
{

/**
 * @brief SGD优化器
 * @details 实现标准随机梯度下降算法
 */
class SGD : public Optimizer
{
private:
    float lr_;            // 学习率
    float momentum_;      // 动量（可选）
    float weight_decay_;  // 权重衰减（可选）
    bool nesterov_;       // Nesterov动量（可选）

    // 状态字典
    std::unordered_map<Parameter *, Tensor> momentum_buffers_;

public:
    /**
     * @brief 构造函数
     * @param target 目标模块
     * @param lr 学习率
     * @param momentum 动量，默认为0
     * @param weight_decay 权重衰减，默认为0
     * @param nesterov 是否使用Nesterov动量，默认为false
     */
    SGD(Module &target, float lr, float momentum = 0.0f, float weight_decay = 0.0f, bool nesterov = false);

protected:
    /**
     * @brief 更新单个参数
     * @param param 参数引用
     */
    void step_one(Parameter &param) override;
};

}  // namespace origin

#endif  // __ORIGIN_DL_SGD_H__
