#include "dlFunction.h"

namespace dl
{

Variable Function::operator()(const Variable &input)
{
    auto x      = input.data;
    auto y      = this->forward(x);
    auto output = Variable(y);
    return output;
}

NdArray Square::forward(const NdArray &x)
{
    return nc::power(x, 2);
}

}  // namespace dl