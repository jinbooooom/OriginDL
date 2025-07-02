#include "dlFunction.h"

namespace dl
{

NdArrayPtrList Add::Forward(const NdArrayPtrList &xs)
{
    // logd("do add");
    auto outputs  = NdArrayPtrList();
    NdArrayPtr x1 = xs[0];
    NdArrayPtr x2 = xs[1];
    auto y        = (*x1) + (*x2);
    outputs.push_back(AsDLArrayPtr(y));

    return outputs;
}

NdArrayPtrList Add::Backward(const NdArrayPtrList &gys)
{
    if (1 != gys.size())
    {
        logw("invalid argument size, not equal to 1");
    }

    auto gxs = NdArrayPtrList{gys[0], gys[0]};

    return gxs;
}

}  // namespace dl
