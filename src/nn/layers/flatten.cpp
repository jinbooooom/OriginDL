#include "origin/nn/layers/flatten.h"
#include "origin/core/operator.h"

namespace origin
{
namespace nn
{

Flatten::Flatten(int start_dim, int end_dim) : start_dim_(start_dim), end_dim_(end_dim) {}

Tensor Flatten::forward(const Tensor &input)
{
    return functional::flatten(input, start_dim_, end_dim_);
}

}  // namespace nn
}  // namespace origin
