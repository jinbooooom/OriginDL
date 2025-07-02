#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Add::Forward(const NdArrayPtrList &xs)
{
    // logd("do add");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x1 = xs[0];
    NdArrayPtr x2 = xs[1];
    auto y        = *x1 + *x2;
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArray Add::Backward(const NdArray &gy)
{
    // auto x  = this->input->data;
    // auto gx = af::exp(x) * gy;
    // return gx;
    return gy;
}

}  // namespace dl
