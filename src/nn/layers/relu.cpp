#include "origin/nn/layers/relu.h"
#include "origin/core/operator.h"

namespace origin
{
namespace nn
{

Tensor ReLU::forward(const Tensor &input)
{
    // 调用 relu 函数
    return relu(input);
}

}  // namespace nn
}  // namespace origin
