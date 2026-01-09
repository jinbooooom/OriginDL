#ifndef __ORIGIN_DL_LOG_H__
#define __ORIGIN_DL_LOG_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief 自然对数算子（以 e 为底的对数）
 *
 * 计算输入张量的自然对数，即 log_e(x) = ln(x)
 * 与 PyTorch 的 torch.log() 行为一致
 */
class Log : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的自然对数（以 e 为底）
 *
 * @param x 输入张量，必须为正数
 * @return 自然对数结果，log_e(x) = ln(x)
 *
 * @note 与 PyTorch 的 torch.log() 行为一致
 */
extern Tensor log(const Tensor &x);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_LOG_H__

