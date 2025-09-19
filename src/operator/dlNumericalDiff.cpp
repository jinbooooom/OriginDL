#include "dlOperator.h"

namespace dl
{

// 数值微分，求函数 f 在 x 处的导数
NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps)
{
    auto x0 = Variable(x.mData - eps);
    auto x1 = Variable(x.mData + eps);
    auto y0 = f(x0);
    auto y1 = f(x1);
    return (y1.mData - y0.mData) / (2 * eps);
}

}  // namespace dl
