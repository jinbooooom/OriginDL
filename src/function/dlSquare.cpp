#include "dlFunction.h"

namespace dl
{

NdArray Square::Forward(const NdArray &x)
{
    return af::pow(x, 2);
}

NdArray Square::Backward(const NdArray &gy)
{
    auto x  = this->input->data;
    auto gx = 2.0 * x * gy;
    return gx;
}

}  // namespace dl
