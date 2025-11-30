#ifndef __ORIGIN_DL_LINEAR_H__
#define __ORIGIN_DL_LINEAR_H__

#include "../../core/operator.h"
#include "../../core/parameter.h"
#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{

/**
 * @brief 全连接层（线性层）
 * @details 实现 y = x * W + b
 */
class Linear : public Layer
{
private:
    Parameter weight_;  // 权重参数
    Parameter bias_;    // 偏置参数
    int in_features_;
    int out_features_;
    bool use_bias_;

public:
    /**
     * @brief 构造函数
     * @param in_features 输入特征数
     * @param out_features 输出特征数
     * @param bias 是否使用偏置，默认为true
     */
    Linear(int in_features, int out_features, bool bias = true);

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 参数访问
     * @return 权重参数
     */
    Parameter *weight() { return &weight_; }

    /**
     * @brief 参数访问
     * @return 偏置参数
     */
    Parameter *bias() { return use_bias_ ? &bias_ : nullptr; }

    /**
     * @brief 重置参数
     */
    void reset_parameters();

private:
    /**
     * @brief 初始化权重参数
     */
    Parameter init_weight();

    /**
     * @brief 初始化偏置参数
     */
    Parameter init_bias();

    /**
     * @brief 初始化参数（用于reset_parameters）
     */
    void init_parameters();
};

}  // namespace origin

#endif  // __ORIGIN_DL_LINEAR_H__
