#include "origin/nn/layers/max_pool2d.h"
#include "origin/core/operator.h"
#include "origin/operators/pooling/max_pool2d.h"

namespace origin
{
namespace nn
{

MaxPool2d::MaxPool2d(std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> pad)
    : kernel_size_(kernel_size), stride_(stride), pad_(pad)
{}

MaxPool2d::MaxPool2d(int kernel_size, int stride, int pad)
    : MaxPool2d({kernel_size, kernel_size}, {stride, stride}, {pad, pad})
{}

Tensor MaxPool2d::forward(const Tensor &input)
{
    // 调用 max_pool2d 函数
    return functional::max_pool2d(input, kernel_size_, stride_, pad_);
}

}  // namespace nn
}  // namespace origin
