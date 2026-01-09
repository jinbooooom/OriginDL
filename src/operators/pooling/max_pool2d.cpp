#include "origin/operators/pooling/max_pool2d.h"
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/conv_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> MaxPool2d::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("MaxPool2d operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x      = xs[0];
    auto x_shape = x.shape();

    // 检查输入形状
    if (x_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("MaxPool2d forward: x must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    // 获取 Mat 引用并调用底层 max_pool2d
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));

    // 清空之前的索引
    indices_.clear();
    auto result = x_mat.max_pool2d(kernel_size_, stride_, pad_, indices_);

    // 保存索引形状
    indices_shape_ = x_shape;  // 保存输入形状，用于验证

    auto y = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> MaxPool2d::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("MaxPool2d backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // 验证索引是否存在
    if (indices_.empty())
    {
        THROW_RUNTIME_ERROR("MaxPool2d backward: indices not found. forward() must be called before backward()");
    }

    // 获取 Mat 引用并调用底层 max_pool2d_backward
    const OriginMat &gy_mat = static_cast<const OriginMat &>(mat(gy));
    const OriginMat &x_mat  = static_cast<const OriginMat &>(mat(x));

    auto gx = x_mat.max_pool2d_backward(gy_mat, kernel_size_, stride_, pad_, indices_);

    std::vector<Tensor> outputs;
    outputs.push_back(convert_mat_to_tensor(std::move(gx)));
    return outputs;
}

Tensor max_pool2d(const Tensor &x, std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> pad)
{
    auto op = std::make_shared<MaxPool2d>(kernel_size, stride, pad);
    return (*op)(x);
}

Tensor max_pool2d(const Tensor &x, int kernel_size, int stride, int pad)
{
    return max_pool2d(x, std::make_pair(kernel_size, kernel_size), std::make_pair(stride, stride),
                      std::make_pair(pad, pad));
}

}  // namespace functional
}  // namespace origin
