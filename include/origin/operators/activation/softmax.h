#ifndef __ORIGIN_DL_SOFTMAX_H__
#define __ORIGIN_DL_SOFTMAX_H__

#include "../../core/operator.h"

namespace origin
{

/**
 * @brief Softmax 算子
 *
 * 计算 softmax 归一化，用于多分类任务
 * 公式：softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * 注意数值稳定性：先减去最大值再计算
 */
class Softmax : public Operator
{
public:
    int axis_;  // 计算 softmax 的轴，默认为 -1（最后一个维度）

    Softmax() : axis_(-1) {}
    Softmax(int axis) : axis_(axis) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的 softmax 归一化
 *
 * @param x 输入张量
 * @param axis 计算 softmax 的轴，默认为 -1（最后一个维度）
 * @return softmax 归一化结果
 */
extern Tensor softmax(const Tensor &x, int axis = -1);

}  // namespace origin

#endif  // __ORIGIN_DL_SOFTMAX_H__

