#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Square::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto o       = af::pow(*x, 2);
    outputs.push_back(AsDLArrayPtr(o));
    return outputs;
}

NdArray Square::Backward(const NdArray &gy)
{
    // auto x  = this->input->data;
    // auto gx = 2.0 * x * gy;
    // return gx;
}

}  // namespace dl
