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

}  // namespace dl