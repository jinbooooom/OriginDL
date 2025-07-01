#include "dlFunction.h"

namespace dl
{

NdArray Exp::Forward(const NdArray &x)
{
    return af::exp(x);
}

NdArray Exp::Backward(const NdArray &gy)
{
    auto x  = this->input->data;
    auto gx = af::exp(x) * gy;
    return gx;
}

}  // namespace dl
