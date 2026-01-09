#ifndef __ORIGIN_DL_CAT_H__
#define __ORIGIN_DL_CAT_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief Cat 拼接算子
 * @details 在指定维度上拼接多个张量
 */
class Cat : public Operator
{
public:
    int dim_;  // 拼接的维度

    explicit Cat(int dim = 0) : dim_(dim) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief Cat 函数
 * @param xs 输入张量列表
 * @param dim 拼接的维度，默认 0
 * @return 拼接后的张量
 */
Tensor cat(const std::vector<Tensor> &xs, int dim = 0);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_CAT_H__

