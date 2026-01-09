#ifndef __ORIGIN_DL_SOFTMAX_CROSS_ENTROPY_H__
#define __ORIGIN_DL_SOFTMAX_CROSS_ENTROPY_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief SoftmaxCrossEntropy 损失函数算子
 *
 * 计算 softmax 交叉熵损失，用于多分类任务
 * 公式：loss = -mean(log(softmax(x)[target]))
 *
 * 输入：
 * - x: (N, C) 形状，N 是 batch size，C 是类别数
 * - target: (N,) 形状，每个元素是类别索引（0 到 C-1）
 *
 * 输出：
 * - loss: 标量
 */
class SoftmaxCrossEntropy : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算 softmax 交叉熵损失
 *
 * @param x 输入 logits，形状为 (N, C)，N 是 batch size，C 是类别数
 * @param target 目标类别索引，形状为 (N,)，每个元素是类别索引（0 到 C-1）
 * @return 交叉熵损失，标量
 */
extern Tensor softmax_cross_entropy(const Tensor &x, const Tensor &target);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_SOFTMAX_CROSS_ENTROPY_H__

