#include "origin/operators/nn/upsample.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"
#include "origin/mat/origin/origin_mat.h"
#include <cmath>

namespace origin
{
namespace functional
{

std::vector<Tensor> Upsample::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Upsample operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];
    auto x_shape = x.shape();

    if (x_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Upsample forward: x must be 4D (N, C, H, W), but got shape {}", 
                           x_shape.to_string());
    }

    // 计算输出形状
    Shape output_shape = x_shape;
    if (size_.first > 0 && size_.second > 0)
    {
        // 使用指定的目标大小
        output_shape[2] = size_.first;   // H
        output_shape[3] = size_.second;  // W
    }
    else
    {
        // 使用缩放因子
        output_shape[2] = static_cast<int>(std::round(x_shape[2] * scale_factor_.first));   // H
        output_shape[3] = static_cast<int>(std::round(x_shape[3] * scale_factor_.second)); // W
    }

    // 简化实现：手动实现最近邻上采样
    // 对于最近邻上采样，我们需要复制每个像素
    
    // 获取输入数据
    auto x_data = x.to_vector<float>();
    
    // 计算缩放因子
    int scale_h = output_shape[2] / x_shape[2];
    int scale_w = output_shape[3] / x_shape[3];
    
    // 创建输出数据
    std::vector<float> output_data(output_shape.elements());
    
    // 最近邻上采样：每个输入像素复制 scale_h * scale_w 次
    int N = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
    int OH = output_shape[2], OW = output_shape[3];
    
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    // 计算对应的输入位置（最近邻）
                    int ih = oh / scale_h;
                    int iw = ow / scale_w;
                    
                    // 计算索引
                    int input_idx = ((n * C + c) * H + ih) * W + iw;
                    int output_idx = ((n * C + c) * OH + oh) * OW + ow;
                    
                    output_data[output_idx] = x_data[input_idx];
                }
            }
        }
    }
    
    auto y = Tensor(output_data, output_shape, x.dtype());
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Upsample::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Upsample backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x = this->inputs_[0];
    
    // 下采样梯度：对每个输入像素，累加所有对应的输出梯度
    auto x_shape = x.shape();
    auto gy_shape = gy.shape();
    
    auto gy_data = gy.to_vector<float>();
    std::vector<float> gx_data(x_shape.elements(), 0.0f);
    
    int scale_h = gy_shape[2] / x_shape[2];
    int scale_w = gy_shape[3] / x_shape[3];
    
    int N = x_shape[0], C = x_shape[1], H = x_shape[2], W = x_shape[3];
    int GY_H = gy_shape[2], GY_W = gy_shape[3];
    
    for (int n = 0; n < N; ++n)
    {
        for (int c = 0; c < C; ++c)
        {
            for (int gy_h = 0; gy_h < GY_H; ++gy_h)
            {
                for (int gy_w = 0; gy_w < GY_W; ++gy_w)
                {
                    int ih = gy_h / scale_h;
                    int iw = gy_w / scale_w;
                    
                    int gy_idx = ((n * C + c) * GY_H + gy_h) * GY_W + gy_w;
                    int gx_idx = ((n * C + c) * H + ih) * W + iw;
                    
                    gx_data[gx_idx] += gy_data[gy_idx];
                }
            }
        }
    }
    
    auto gx = Tensor(gx_data, x_shape, gy.dtype());
    return std::vector<Tensor>{std::move(gx)};
}

Tensor upsample(const Tensor &x, const std::string &mode, std::pair<float, float> scale_factor)
{
    auto op = std::make_shared<Upsample>(mode, scale_factor);
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin

