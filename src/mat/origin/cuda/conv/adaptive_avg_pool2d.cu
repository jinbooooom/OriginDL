#include <cuda_runtime.h>
#include <cstring>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/exception.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace cuda
{

// ==================== CUDA Kernels ====================

/**
 * @brief 自适应平均池化前向传播的CUDA kernel
 */
template <typename T>
__global__ void adaptive_avg_pool2d_forward_kernel(const T *__restrict__ x,
                                                   T *__restrict__ y,
                                                   size_t N,
                                                   size_t C,
                                                   size_t H_in,
                                                   size_t W_in,
                                                   int H_out,
                                                   int W_out,
                                                   int kernel_h,
                                                   int kernel_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H_out * W_out;

    if (idx < total_elements)
    {
        // 计算输出位置 (n, c, h_out, w_out)
        size_t n     = idx / (C * H_out * W_out);
        size_t rem   = idx % (C * H_out * W_out);
        size_t c     = rem / (H_out * W_out);
        rem          = rem % (H_out * W_out);
        int h_out    = rem / W_out;
        int w_out    = rem % W_out;

        // 计算输入区域的起始和结束位置
        int h_start = h_out * kernel_h;
        int h_end   = min(h_start + kernel_h, static_cast<int>(H_in));
        int w_start = w_out * kernel_w;
        int w_end   = min(w_start + kernel_w, static_cast<int>(W_in));

        // 计算平均值
        T sum   = T(0);
        int count = 0;

        for (int h = h_start; h < h_end; ++h)
        {
            for (int w = w_start; w < w_end; ++w)
            {
                // 行主序索引：n * (C*H*W) + c * (H*W) + h * W + w
                size_t x_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + static_cast<size_t>(h) * W_in +
                                static_cast<size_t>(w);
                sum += x[x_idx];
                count++;
            }
        }

        T avg = (count > 0) ? sum / static_cast<T>(count) : T(0);
        y[idx] = avg;
    }
}

/**
 * @brief 自适应平均池化反向传播的CUDA kernel
 */
template <typename T>
__global__ void adaptive_avg_pool2d_backward_kernel(const T *__restrict__ gy,
                                                     T *__restrict__ gx,
                                                     size_t N,
                                                     size_t C,
                                                     size_t H_in,
                                                     size_t W_in,
                                                     size_t H_out,
                                                     size_t W_out,
                                                     int kernel_h,
                                                     int kernel_w)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H_out * W_out;

    if (idx < total_elements)
    {
        // 计算输出位置 (n, c, h_out, w_out)
        size_t n     = idx / (C * H_out * W_out);
        size_t rem   = idx % (C * H_out * W_out);
        size_t c     = rem / (H_out * W_out);
        rem          = rem % (H_out * W_out);
        size_t h_out = rem / W_out;
        size_t w_out = rem % W_out;

        // 计算输入区域的起始和结束位置
        int h_start = static_cast<int>(h_out) * kernel_h;
        int h_end   = min(h_start + kernel_h, static_cast<int>(H_in));
        int w_start = static_cast<int>(w_out) * kernel_w;
        int w_end   = min(w_start + kernel_w, static_cast<int>(W_in));

        int count = (h_end - h_start) * (w_end - w_start);
        T grad_value = gy[idx];
        T grad_per_element = (count > 0) ? grad_value / static_cast<T>(count) : T(0);

                // 将梯度分配到输入区域
        // 注意：atomicAdd 只支持 float, double, int32, uint32, uint64
        // 对于其他类型，使用类型特化
        for (int h = h_start; h < h_end; ++h)
        {
            for (int w = w_start; w < w_end; ++w)
            {
                size_t gx_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + static_cast<size_t>(h) * W_in +
                                 static_cast<size_t>(w);
                if constexpr (std::is_same_v<T, float>)
                {
                    atomicAdd(reinterpret_cast<float *>(&gx[gx_idx]), static_cast<float>(grad_per_element));
                }
                else if constexpr (std::is_same_v<T, double>)
                {
                    atomicAdd(reinterpret_cast<double *>(&gx[gx_idx]), static_cast<double>(grad_per_element));
                }
                else
                {
                    // 对于其他类型，不支持自适应池化的反向传播
                    // 这应该在类型分发时被过滤掉
                    gx[gx_idx] += grad_per_element;  // 注意：这可能导致竞争条件
                }
            }
        }
    }
}

// ==================== adaptive_avg_pool2d 实现 ====================

std::unique_ptr<Mat> adaptive_avg_pool2d(const OriginMat &x, std::pair<int, int> output_size)
{
    // 输入验证：确保输入是4D张量 (N, C, H, W)
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("adaptive_avg_pool2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }

    VALIDATE_CUDA_DEVICE(x);

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
    int kernel_h = static_cast<int>(H_in) / H_out;
    int kernel_w = static_cast<int>(W_in) / W_out;

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

        size_t total_elements = N * C * H_out * W_out;
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        adaptive_avg_pool2d_forward_kernel<T><<<num_blocks, threads_per_block>>>(x_ptr, y_ptr, N, C, H_in, W_in, H_out,
                                                                                 W_out, kernel_h, kernel_w);
    });

    CUDA_CHECK_ASYNC();

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

    VALIDATE_CUDA_DEVICE(x);

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

    // 初始化梯度为0
    cudaMemset(gx_data, 0, x.shape().elements() * element_size(gy.dtype()));
    CUDA_CHECK_ASYNC();

    // 使用类型分发器执行计算（只支持浮点类型）
    if (gy.dtype() == DataType::kFloat32)
    {
        const float *gy_ptr = static_cast<const float *>(gy_data);
        float *gx_ptr       = static_cast<float *>(gx_data);

        size_t total_elements = N * C * H_out * W_out;
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        adaptive_avg_pool2d_backward_kernel<float><<<num_blocks, threads_per_block>>>(gy_ptr, gx_ptr, N, C, H_in, W_in,
                                                                                       H_out, W_out, kernel_h, kernel_w);
    }
    else if (gy.dtype() == DataType::kFloat64)
    {
        const double *gy_ptr = static_cast<const double *>(gy_data);
        double *gx_ptr        = static_cast<double *>(gx_data);

        size_t total_elements = N * C * H_out * W_out;
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        adaptive_avg_pool2d_backward_kernel<double><<<num_blocks, threads_per_block>>>(gy_ptr, gx_ptr, N, C, H_in, W_in,
                                                                                        H_out, W_out, kernel_h, kernel_w);
    }
    else
    {
        THROW_INVALID_ARG("adaptive_avg_pool2d_backward: only float32 and float64 are supported, got {}",
                          dtype_to_string(gy.dtype()));
    }

    CUDA_CHECK_ASYNC();

    return result;
}

}  // namespace cuda
}  // namespace origin

