#ifndef __ORIGIN_DL_COMPARE_H__
#define __ORIGIN_DL_COMPARE_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief 等于比较算子
 */
class Equal : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 不等于比较算子
 */
class NotEqual : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 小于比较算子
 */
class Less : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 小于等于比较算子
 */
class LessEqual : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 大于比较算子
 */
class Greater : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 大于等于比较算子
 */
class GreaterEqual : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

}  // namespace functional

// 比较运算符（对张量）
Tensor operator==(const Tensor &lhs, const Tensor &rhs);
Tensor operator!=(const Tensor &lhs, const Tensor &rhs);
Tensor operator<(const Tensor &lhs, const Tensor &rhs);
Tensor operator<=(const Tensor &lhs, const Tensor &rhs);
Tensor operator>(const Tensor &lhs, const Tensor &rhs);
Tensor operator>=(const Tensor &lhs, const Tensor &rhs);

// 比较运算符（对标量）
Tensor operator==(const Tensor &lhs, const Scalar &rhs);
Tensor operator!=(const Tensor &lhs, const Scalar &rhs);
Tensor operator<(const Tensor &lhs, const Scalar &rhs);
Tensor operator<=(const Tensor &lhs, const Scalar &rhs);
Tensor operator>(const Tensor &lhs, const Scalar &rhs);
Tensor operator>=(const Tensor &lhs, const Scalar &rhs);

}  // namespace origin

#endif  // __ORIGIN_DL_COMPARE_H__
