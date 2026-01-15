#include <random>
#include <vector>
#include "origin/core/operator.h"
#include "origin/mat/mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

Dropout::Dropout(float p, bool training) : p_(p), training_(training)
{
    if (p < 0.0f || p >= 1.0f)
    {
        THROW_INVALID_ARG("Dropout: p must be in [0, 1), but got {}", p);
    }
}

std::vector<Tensor> Dropout::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Dropout operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];

    if (!training_)
    {
        // 推理模式：直接返回输入
        std::vector<Tensor> outputs;
        outputs.push_back(x);
        return outputs;
    }

    // 训练模式：生成 dropout mask
    auto x_shape = x.shape();
    auto x_data  = x.to_vector<float>();

    // 生成随机 mask：值为 0 或 1/(1-p)
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> mask_data(x_data.size());
    std::vector<float> y_data(x_data.size());

    float scale = 1.0f / (1.0f - p_);

    for (size_t i = 0; i < x_data.size(); ++i)
    {
        if (dist(gen) < p_)
        {
            mask_data[i] = 0.0f;
            y_data[i]    = 0.0f;
        }
        else
        {
            mask_data[i] = scale;
            y_data[i]    = x_data[i] * scale;
        }
    }

    // 保存 mask 用于反向传播
    mask_ = Tensor(mask_data, x_shape, dtype(DataType::kFloat32).device(x.device()));

    auto y = Tensor(y_data, x_shape, dtype(DataType::kFloat32).device(x.device()));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Dropout::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Dropout backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];

    if (!training_)
    {
        // 推理模式：梯度直接传递
        return std::vector<Tensor>{std::move(gy)};
    }

    // 训练模式：根据 mask 计算梯度
    auto gy_data   = gy.to_vector<float>();
    auto mask_data = mask_.to_vector<float>();
    auto gy_shape  = gy.shape();

    std::vector<float> gx_data(gy_data.size());
    for (size_t i = 0; i < gy_data.size(); ++i)
    {
        gx_data[i] = gy_data[i] * mask_data[i];
    }

    auto gx = Tensor(gx_data, gy_shape, dtype(DataType::kFloat32).device(gy.device()));
    return std::vector<Tensor>{std::move(gx)};
}

}  // namespace functional
}  // namespace origin
