#include <algorithm>
#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> ReLU::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("ReLU operator requires exactly 1 input, but got {}", xs.size());
    }

    // ReLU: y = max(0, x)
    // 使用 max(0, x) = x * (x > 0) + 0 * (x <= 0)
    // 由于没有直接的 max 或比较操作，我们需要手动实现
    // 为了简化，我们先在算子层面直接计算

    auto &x      = xs[0];
    auto x_data  = x.to_vector<float>();
    auto x_shape = x.shape();

    // 计算 ReLU：y = max(0, x)
    std::vector<float> y_data(x_data.size());
    for (size_t i = 0; i < x_data.size(); ++i)
    {
        y_data[i] = std::max(0.0f, x_data[i]);
    }

    auto result = Tensor(y_data, x_shape, dtype(DataType::kFloat32).device(x.device()));

    std::vector<Tensor> outputs;
    outputs.push_back(result);
    return outputs;
}

std::vector<Tensor> ReLU::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("ReLU backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // ReLU 的梯度：gx = gy * (x > 0 ? 1 : 0)
    auto &x  = this->inputs_[0];
    auto &gy = gys[0];

    auto x_data  = x.to_vector<float>();
    auto gy_data = gy.to_vector<float>();
    auto x_shape = x.shape();

    // 计算梯度：gx = gy * (x > 0)
    std::vector<float> gx_data(x_data.size());
    for (size_t i = 0; i < x_data.size(); ++i)
    {
        gx_data[i] = (x_data[i] > 0.0f) ? gy_data[i] : 0.0f;
    }

    auto gx = Tensor(gx_data, x_shape, dtype(DataType::kFloat32).device(x.device()));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor relu(const Tensor &x)
{
    auto op = std::make_shared<ReLU>();
    return (*op)(x);
}

}  // namespace origin
