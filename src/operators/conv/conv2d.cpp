#include "origin/operators/conv/conv2d.h"
#include "origin/utils/exception.h"
#include "origin/core/tensor.h"
#include "origin/core/operator.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/conv_utils.h"
#include <vector>

namespace origin
{

std::vector<Tensor> Conv2d::forward(const std::vector<Tensor> &xs)
{
    // xs[0] = x (输入), xs[1] = W (卷积核), xs[2] = b (偏置，可选)
    if (xs.size() < 2 || xs.size() > 3)
    {
        THROW_RUNTIME_ERROR("Conv2d operator requires 2 or 3 inputs (x, W, [b]), but got {}", xs.size());
    }

    auto &x = xs[0];
    auto &W = xs[1];
    const Tensor *b = (xs.size() == 3) ? &xs[2] : nullptr;

    auto x_shape = x.shape();
    auto W_shape = W.shape();

    // 检查输入形状
    if (x_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: x must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    if (W_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: W must be 4D (OC, C, KH, KW), but got shape {}", W_shape.to_string());
    }

    size_t C = x_shape[1];
    size_t C_in = W_shape[1];

    // 检查通道数是否匹配
    if (C != C_in)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: channel mismatch - x has {} channels, but W expects {} channels", C,
                           C_in);
    }

    // 获取 Mat 引用并调用底层 conv2d（所有实现细节都在底层完成）
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));
    const OriginMat &W_mat = static_cast<const OriginMat &>(mat(W));
    const OriginMat *b_mat = (b != nullptr) ? &static_cast<const OriginMat &>(mat(*b)) : nullptr;

    auto result = x_mat.conv2d(W_mat, b_mat, stride_, pad_);
    auto y = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> Conv2d::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Conv2d backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x = this->inputs_[0];
    auto &W = this->inputs_[1];
    const Tensor *b = (this->inputs_.size() == 3) ? &this->inputs_[2] : nullptr;

    // 获取 Mat 引用并调用底层 conv2d_backward（所有实现细节都在底层完成）
    const OriginMat &gy_mat = static_cast<const OriginMat &>(mat(gy));
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));
    const OriginMat &W_mat = static_cast<const OriginMat &>(mat(W));
    const OriginMat *b_mat = (b != nullptr) ? &static_cast<const OriginMat &>(mat(*b)) : nullptr;

    auto grads = gy_mat.conv2d_backward(gy_mat, x_mat, W_mat, b_mat, stride_, pad_);

    std::vector<Tensor> outputs;
    // grads 顺序为 {gx, gW, [gb]}
    outputs.push_back(convert_mat_to_tensor(std::move(grads[0])));
    outputs.push_back(convert_mat_to_tensor(std::move(grads[1])));
    if (b != nullptr && grads.size() > 2)
    {
        outputs.push_back(convert_mat_to_tensor(std::move(grads[2])));
    }
    return outputs;
}

Tensor conv2d(const Tensor &x, const Tensor &W, const Tensor *b, std::pair<int, int> stride, std::pair<int, int> pad)
{
    auto op = std::make_shared<Conv2d>(stride, pad);
    if (b != nullptr)
    {
        return (*op)({x, W, *b})[0];
    }
    else
    {
        return (*op)({x, W})[0];
    }
}

Tensor conv2d(const Tensor &x, const Tensor &W, const Tensor *b, int stride, int pad)
{
    return conv2d(x, W, b, pair(stride), pair(pad));
}

}  // namespace origin

