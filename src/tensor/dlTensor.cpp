#include "base/dlException.h"
#include "dlTensor.h"

namespace dl
{

// 工厂函数
TensorPtr make_tensor(const NdArray &data) {
    return std::make_shared<Tensor>(data);
}

TensorPtr make_tensor(NdArray &&data) {
    return std::make_shared<Tensor>(std::move(data));
}


}  // namespace dl