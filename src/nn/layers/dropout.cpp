#include "origin/nn/layers/dropout.h"
#include "origin/core/operator.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace nn
{

Dropout::Dropout(float p) : p_(p)
{
    if (p < 0.0f || p >= 1.0f)
    {
        THROW_INVALID_ARG("Dropout: p must be in [0, 1), but got {}", p);
    }
}

Tensor Dropout::forward(const Tensor &input)
{
    // 创建 Dropout Operator（注意：Operator 的 Dropout 需要两个参数）
    // 使用完整的命名空间来区分 Layer 和 Operator
    auto op = std::make_shared<::origin::Dropout>(p_, is_training());

    // 执行前向传播
    std::vector<Tensor> inputs;
    inputs.push_back(input);
    auto outputs = (*op)(inputs);

    return outputs[0];
}

}  // namespace nn
}  // namespace origin

