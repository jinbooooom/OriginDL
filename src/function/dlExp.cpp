#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Exp::Forward(const NdArrayPtrList &xs)
{
    auto outputs = NdArrayPtrList();
    NdArrayPtr x = xs[0];
    auto o       = af::exp(*x);
    outputs.push_back(AsDLArrayPtr(o));
    return outputs;
}

NdArray Exp::Backward(const NdArray &gy)
{
    // auto x  = this->input->data;
    // auto gx = af::exp(x) * gy;
    // return gx;
}

}  // namespace dl
