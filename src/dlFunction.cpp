#include "dlFunction.h"

namespace dl
{

Variable Function::operator()(const Variable &input)
{
    auto x      = input.data;
    auto y      = this->Forward(x);
    auto output = Variable(y);
    return output;
}

NdArray Square::Forward(const NdArray &x)
{
    return nc::power(x, 2);
}

NdArray Exp::Forward(const NdArray &x)
{
    return nc::exp(x);
}

// 数值微分，求函数 f 在 x 处的导数
NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps)
{
    auto x0 = Variable(x.data - eps);
    auto x1 = Variable(x.data + eps);
    auto y0 = f(x0);
    auto y1 = f(x1);
    return (y1.data - y0.data) / (2 * eps);
}

}  // namespace dl