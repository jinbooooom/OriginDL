#include "origin/operators/conv/adaptive_avg_pool2d.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"
#include "origin/mat/origin/origin_mat.h"

namespace origin
{

std::vector<Tensor> AdaptiveAvgPool2d::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("AdaptiveAvgPool2d operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];
    auto x_shape = x.shape();

    // 检查输入形状：应该是 (N, C, H, W)
    if (x_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("AdaptiveAvgPool2d forward: x must be 4D (N, C, H, W), but got shape {}", 
                           x_shape.to_string());
    }

    size_t N = x_shape[0];
    size_t C = x_shape[1];
    size_t H_in = x_shape[2];
    size_t W_in = x_shape[3];
    
    int H_out = output_size_.first;
    int W_out = output_size_.second;

    // 计算池化窗口大小
    // 自适应池化：将输入区域平均分成输出尺寸的网格
    int kernel_h = static_cast<int>(H_in) / H_out;
    int kernel_w = static_cast<int>(W_in) / W_out;
    
    // 如果无法整除，使用向上取整
    if (H_in % H_out != 0)
    {
        kernel_h = (static_cast<int>(H_in) + H_out - 1) / H_out;
    }
    if (W_in % W_out != 0)
    {
        kernel_w = (static_cast<int>(W_in) + W_out - 1) / W_out;
    }

    // 对于每个输出位置，计算对应输入区域的平均值
    // 注意：当前使用 CPU 实现，后续可以优化为 GPU
    // 由于没有直接的 adaptive pool 实现，我们使用 reshape + average pool 的方式
    // 或者直接实现平均池化逻辑
    
    // 创建输出形状
    Shape output_shape{N, C, static_cast<size_t>(H_out), static_cast<size_t>(W_out)};
    
    // 使用 CPU 实现（后续可以优化为 GPU）
    auto x_data = x.to_vector<float>();
    std::vector<float> output_data(output_shape.elements());
    
    // 对每个输出位置计算平均值
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (int h_out = 0; h_out < H_out; ++h_out)
            {
                for (int w_out = 0; w_out < W_out; ++w_out)
                {
                    // 计算输入区域的起始和结束位置
                    int h_start = h_out * kernel_h;
                    int h_end = std::min(h_start + kernel_h, static_cast<int>(H_in));
                    int w_start = w_out * kernel_w;
                    int w_end = std::min(w_start + kernel_w, static_cast<int>(W_in));
                    
                    // 计算平均值
                    float sum = 0.0f;
                    int count = 0;
                    
                    for (int h = h_start; h < h_end; ++h)
                    {
                        for (int w = w_start; w < w_end; ++w)
                        {
                            // 行主序索引：n * (C*H*W) + c * (H*W) + h * W + w
                            size_t idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h * W_in + w;
                            sum += x_data[idx];
                            count++;
                        }
                    }
                    
                    float avg = (count > 0) ? sum / count : 0.0f;
                    
                    // 输出索引：n * (C*H_out*W_out) + c * (H_out*W_out) + h_out * W_out + w_out
                    size_t out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out;
                    output_data[out_idx] = avg;
                }
            }
        }
    }
    
    // 创建输出 tensor，保持原始设备
    auto y = Tensor(output_data, output_shape, dtype(x.dtype()).device(x.device()));
    
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> AdaptiveAvgPool2d::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("AdaptiveAvgPool2d backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x = this->inputs_[0];
    
    auto x_shape = x.shape();
    auto gy_shape = gy.shape();
    
    size_t N = x_shape[0];
    size_t C = x_shape[1];
    size_t H_in = x_shape[2];
    size_t W_in = x_shape[3];
    size_t H_out = gy_shape[2];
    size_t W_out = gy_shape[3];
    
    // 计算池化窗口大小
    int kernel_h = static_cast<int>(H_in) / static_cast<int>(H_out);
    int kernel_w = static_cast<int>(W_in) / static_cast<int>(W_out);
    
    if (H_in % H_out != 0)
    {
        kernel_h = (static_cast<int>(H_in) + static_cast<int>(H_out) - 1) / static_cast<int>(H_out);
    }
    if (W_in % W_out != 0)
    {
        kernel_w = (static_cast<int>(W_in) + static_cast<int>(W_out) - 1) / static_cast<int>(W_out);
    }
    
    auto gy_data = gy.to_vector<float>();
    std::vector<float> gx_data(x_shape.elements(), 0.0f);
    
    // 将梯度平均分配到输入区域
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h_out = 0; h_out < H_out; ++h_out)
            {
                for (size_t w_out = 0; w_out < W_out; ++w_out)
                {
                    int h_start = static_cast<int>(h_out) * kernel_h;
                    int h_end = std::min(h_start + kernel_h, static_cast<int>(H_in));
                    int w_start = static_cast<int>(w_out) * kernel_w;
                    int w_end = std::min(w_start + kernel_w, static_cast<int>(W_in));
                    
                    int count = (h_end - h_start) * (w_end - w_start);
                    float grad_value = gy_data[n * (C * H_out * W_out) + c * (H_out * W_out) + 
                                               h_out * W_out + w_out];
                    float grad_per_element = (count > 0) ? grad_value / count : 0.0f;
                    
                    for (int h = h_start; h < h_end; ++h)
                    {
                        for (int w = w_start; w < w_end; ++w)
                        {
                            size_t idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h * W_in + w;
                            gx_data[idx] += grad_per_element;
                        }
                    }
                }
            }
        }
    }
    
    auto gx = Tensor(gx_data, x_shape, dtype(gy.dtype()).device(gy.device()));
    
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

}  // namespace origin

