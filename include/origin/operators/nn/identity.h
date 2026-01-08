#ifndef __ORIGIN_DL_IDENTITY_H__
#define __ORIGIN_DL_IDENTITY_H__

#include "../../core/operator.h"

namespace origin
{

/**
 * @brief Identity 算子：恒等映射
 * @details 直接返回输入，不做任何变换
 */
class Identity : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief Identity 函数
 * @param x 输入张量
 * @return 输出张量（与输入相同）
 */
Tensor identity(const Tensor &x);

}  // namespace origin

#endif  // __ORIGIN_DL_IDENTITY_H__

