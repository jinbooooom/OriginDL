#include <algorithm>
#include <cstring>
#include <vector>
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

// ==================== adaptive_avg_pool2d 实现 ====================

std::unique_ptr<Mat> adaptive_avg_pool2d(const OriginMat &x, std::pair<int, int> output_size)
{
    // 输入验证：确保输入是4D张量 (N, C, H, W)
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("adaptive_avg_pool2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }

    size_t N    = x.shape()[0];
    size_t C    = x.shape()[1];
    size_t H_in = x.shape()[2];
    size_t W_in = x.shape()[3];

    int H_out = output_size.first;
    int W_out = output_size.second;

    if (H_out <= 0 || W_out <= 0)
    {
        THROW_INVALID_ARG("adaptive_avg_pool2d: output_size must be positive, got ({}, {})", H_out, W_out);
    }

    // 计算池化窗口大小
    // 自适应池化：将输入区域平均分成输出尺寸的网格
    int kernel_h = static_cast<int>(H_in) / H_out;
    int kernel_w = static_cast<int>(W_in) / W_out;

    // 如果无法整除，使用向上取整
    if (H_in % static_cast<size_t>(H_out) != 0)
    {
        kernel_h = (static_cast<int>(H_in) + H_out - 1) / H_out;
    }
    if (W_in % static_cast<size_t>(W_out) != 0)
    {
        kernel_w = (static_cast<int>(W_in) + W_out - 1) / W_out;
    }

    // 创建输出形状
    Shape output_shape{N, C, static_cast<size_t>(H_out), static_cast<size_t>(W_out)};
    auto result = std::make_unique<OriginMat>(output_shape, x.dtype(), x.device());

    // 获取数据指针
    const void *x_data = x.storage()->data();
    void *y_data       = result->storage()->data();

    // 使用类型分发器执行计算
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *x_ptr = static_cast<const T *>(x_data);
        T *y_ptr       = static_cast<T *>(y_data);

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
                        int h_end   = std::min(h_start + kernel_h, static_cast<int>(H_in));
                        int w_start = w_out * kernel_w;
                        int w_end   = std::min(w_start + kernel_w, static_cast<int>(W_in));

                        // 计算平均值
                        T sum     = T(0);
                        int count = 0;

                        for (int h = h_start; h < h_end; ++h)
                        {
                            for (int w = w_start; w < w_end; ++w)
                            {
                                // 行主序索引：n * (C*H*W) + c * (H*W) + h * W + w
                                size_t idx = n * (C * H_in * W_in) + c * (H_in * W_in) + static_cast<size_t>(h) * W_in +
                                             static_cast<size_t>(w);
                                sum += x_ptr[idx];
                                count++;
                            }
                        }

                        T avg = (count > 0) ? sum / static_cast<T>(count) : T(0);

                        // 输出索引：n * (C*H_out*W_out) + c * (H_out*W_out) + h_out * W_out + w_out
                        size_t out_idx = n * (C * H_out * W_out) + c * (H_out * W_out) +
                                         static_cast<size_t>(h_out) * W_out + static_cast<size_t>(w_out);
                        y_ptr[out_idx] = avg;
                    }
                }
            }
        }
    });

    return result;
}

std::unique_ptr<Mat> adaptive_avg_pool2d_backward(const OriginMat &gy,
                                                  const OriginMat &x,
                                                  std::pair<int, int> output_size)
{
    // 输入验证：确保 gy 形状为 (N, C, OH, OW)
    if (gy.shape().size() != 4)
    {
        THROW_INVALID_ARG("adaptive_avg_pool2d_backward: gy must be 4D (N, C, OH, OW), but got shape {}",
                          gy.shape().to_string());
    }

    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("adaptive_avg_pool2d_backward: x must be 4D (N, C, H, W), but got shape {}",
                          x.shape().to_string());
    }

    size_t N     = x.shape()[0];
    size_t C     = x.shape()[1];
    size_t H_in  = x.shape()[2];
    size_t W_in  = x.shape()[3];
    size_t H_out = gy.shape()[2];
    size_t W_out = gy.shape()[3];

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

    // 创建输出梯度
    auto result = std::make_unique<OriginMat>(x.shape(), gy.dtype(), gy.device());

    // 获取数据指针
    const void *gy_data = gy.storage()->data();
    void *gx_data       = result->storage()->data();

    // 使用类型分发器执行计算
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_ptr = static_cast<const T *>(gy_data);
        T *gx_ptr       = static_cast<T *>(gx_data);

        // 初始化梯度为0
        std::fill(gx_ptr, gx_ptr + x.shape().elements(), T(0));

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
                        int h_end   = std::min(h_start + kernel_h, static_cast<int>(H_in));
                        int w_start = static_cast<int>(w_out) * kernel_w;
                        int w_end   = std::min(w_start + kernel_w, static_cast<int>(W_in));

                        int count    = (h_end - h_start) * (w_end - w_start);
                        T grad_value = gy_ptr[n * (C * H_out * W_out) + c * (H_out * W_out) + h_out * W_out + w_out];
                        T grad_per_element = (count > 0) ? grad_value / static_cast<T>(count) : T(0);

                        for (int h = h_start; h < h_end; ++h)
                        {
                            for (int w = w_start; w < w_end; ++w)
                            {
                                size_t idx = n * (C * H_in * W_in) + c * (H_in * W_in) + static_cast<size_t>(h) * W_in +
                                             static_cast<size_t>(w);
                                gx_ptr[idx] += grad_per_element;
                            }
                        }
                    }
                }
            }
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
