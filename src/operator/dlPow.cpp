#include "dlOperator.h"

namespace dl
{

NdArrayPtrList Pow::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto &base   = *x;

    DLMat result;
    if (0 == mExponent)
    {
        // 创建全1数组（排除底数为0的元素）
        result = af::constant(1, base.dims(), base.type());
        // 0^0 未定义
        // 生成一个与 base 数组同尺寸的 布尔掩码数组。
        // 若 base 中某元素为 0，则掩码对应位置为 true；否则为 false
        // 通过掩码筛选 result 数组中值为 0 的位置，设置为 Nan
        result(base == 0) = af::NaN;
        outputs.push_back(AsDLArrayPtr(result));
    }
    else if (mExponent > 0)
    {
        result = af::pow(base, mExponent);  // 正整指数
    }
    else
    {
        // 负整指数：先计算正幂再取倒数
        result = 1 / af::pow(base, -mExponent);
    }
    // 如果考虑性能，当 mExponent 为浮点数时，另外讨论情况

    outputs.push_back(AsDLArrayPtr(result));
    return outputs;
}

NdArrayPtrList Pow::Backward(const NdArrayPtrList &gys)
{
    auto x = this->inputs[0]->mData;
    // TODO：考虑 mExponent 为负的情况，暂时没有这个场景
    auto gx = mExponent * af::pow(x, mExponent - 1) * (*gys[0]);
    return AsDLArrayPtrList(gx);
}

VariablePtr pow(const VariablePtr &base, int exponent)
{
    auto f = std::shared_ptr<Operator>(new Pow(exponent));
    auto y = (*f)(base);
    return y[0];
}

VariablePtr operator^(const VariablePtr &base, int exponent)
{
    return pow(base, exponent);
}

}  // namespace dl
