#include <cuda_runtime.h>
#include <cmath>
#include <memory>
#include <type_traits>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

// 本文件仅实现最近邻上采样；mode 参数保留但未使用，双线性未实现。

/**
 * @brief CUDA upsample kernel（最近邻上采样）
 * @details 每个线程处理一个输出元素
 */
template <typename T>
__global__ void upsample_kernel(const T *__restrict__ x,
                                T *__restrict__ y,
                                int N,
                                int C,
                                int H,
                                int W,
                                int OH,
                                int OW,
                                int scale_h,
                                int scale_w)
{
    // 计算输出索引
    int idx            = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * OH * OW;

    if (idx < total_elements)
    {
        // 将线性索引转换为 (n, c, oh, ow)
        int n         = idx / (C * OH * OW);
        int remainder = idx % (C * OH * OW);
        int c         = remainder / (OH * OW);
        remainder     = remainder % (OH * OW);
        int oh        = remainder / OW;
        int ow        = remainder % OW;

        // 计算对应的输入位置（最近邻）
        int ih = oh / scale_h;
        int iw = ow / scale_w;

        // 计算输入索引
        int input_idx  = ((n * C + c) * H + ih) * W + iw;
        int output_idx = ((n * C + c) * OH + oh) * OW + ow;

        y[output_idx] = x[input_idx];
    }
}

/**
 * @brief CUDA upsample_backward kernel（最近邻上采样反向传播）
 * @details 每个线程处理一个输出梯度元素，累加到对应的输入梯度位置
 */
template <typename T>
__global__ void upsample_backward_kernel(const T *__restrict__ gy,
                                         T *__restrict__ gx,
                                         int N,
                                         int C,
                                         int H,
                                         int W,
                                         int GY_H,
                                         int GY_W,
                                         int scale_h,
                                         int scale_w)
{
    // 计算输出梯度索引
    int idx            = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * GY_H * GY_W;

    if (idx < total_elements)
    {
        // 将线性索引转换为 (n, c, gy_h, gy_w)
        int n         = idx / (C * GY_H * GY_W);
        int remainder = idx % (C * GY_H * GY_W);
        int c         = remainder / (GY_H * GY_W);
        remainder     = remainder % (GY_H * GY_W);
        int gy_h      = remainder / GY_W;
        int gy_w      = remainder % GY_W;

        // 计算对应的输入位置
        int ih = gy_h / scale_h;
        int iw = gy_w / scale_w;

        // 计算索引并使用原子操作累加（因为多个输出梯度可能映射到同一个输入位置）
        int gy_idx = ((n * C + c) * GY_H + gy_h) * GY_W + gy_w;
        int gx_idx = ((n * C + c) * H + ih) * W + iw;

        // atomicAdd 只支持 float, double, int32, uint32, uint64
        // 对于其他类型，使用类型特化
        if constexpr (std::is_same_v<T, float>)
        {
            atomicAdd(reinterpret_cast<float *>(&gx[gx_idx]), static_cast<float>(gy[gy_idx]));
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            atomicAdd(reinterpret_cast<double *>(&gx[gx_idx]), static_cast<double>(gy[gy_idx]));
        }
        else if constexpr (std::is_same_v<T, int32_t>)
        {
            atomicAdd(reinterpret_cast<int *>(&gx[gx_idx]), static_cast<int>(gy[gy_idx]));
        }
        else if constexpr (std::is_same_v<T, uint32_t>)
        {
            atomicAdd(reinterpret_cast<unsigned int *>(&gx[gx_idx]), static_cast<unsigned int>(gy[gy_idx]));
        }
        else if constexpr (std::is_same_v<T, uint64_t>)
        {
            atomicAdd(reinterpret_cast<unsigned long long *>(&gx[gx_idx]), static_cast<unsigned long long>(gy[gy_idx]));
        }
        else
        {
            // 对于其他类型，转换为 float 进行原子操作
            float *gx_float = reinterpret_cast<float *>(&gx[gx_idx]);
            float val       = static_cast<float>(gy[gy_idx]);
            atomicAdd(gx_float, val);
        }
    }
}

/**
 * @brief CUDA upsample：上采样操作（最近邻或双线性）
 * @param x 输入张量 (N, C, H, W)
 * @param output_shape 输出形状 (N, C, OH, OW)
 * @param scale_h 高度缩放因子
 * @param scale_w 宽度缩放因子
 * @param mode "nearest" 或 "bilinear"
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> upsample(const OriginMat &x,
                              const Shape &output_shape,
                              int scale_h,
                              int scale_w,
                              const std::string &mode)
{
    if (unlikely(x.shape().size() != 4))
    {
        THROW_INVALID_ARG("Upsample: input must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }

    if (unlikely(output_shape.size() != 4))
    {
        THROW_INVALID_ARG("Upsample: output_shape must be 4D (N, C, OH, OW), but got shape {}",
                          output_shape.to_string());
    };

    auto x_shape = x.shape();
    int N        = x_shape[0];
    int C        = x_shape[1];
    int H        = x_shape[2];
    int W        = x_shape[3];
    int OH       = output_shape[2];
    int OW       = output_shape[3];

    auto result        = std::make_unique<OriginMat>(output_shape, x.dtype(), x.device());
    const void *x_data = x.storage()->data();
    void *y_data       = result->storage()->data();

    const size_t threads_per_block = 256;
    const size_t num_elements      = output_shape.elements();
    const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

    (void)mode;  // 保留接口，当前仅实现最近邻
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        upsample_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(x_data), static_cast<T *>(y_data),
                                                              N, C, H, W, OH, OW, scale_h, scale_w);
    });

    return result;
}

/**
 * @brief CUDA upsample_backward：上采样反向传播
 */
std::unique_ptr<Mat> upsample_backward(const OriginMat &gy,
                                       const Shape &x_shape,
                                       int scale_h,
                                       int scale_w,
                                       const std::string &mode)
{
    if (unlikely(gy.shape().size() != 4))
    {
        THROW_INVALID_ARG("Upsample backward: gradient must be 4D (N, C, OH, OW), but got shape {}",
                          gy.shape().to_string());
    }

    if (unlikely(x_shape.size() != 4))
    {
        THROW_INVALID_ARG("Upsample backward: x_shape must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    auto gy_shape = gy.shape();
    int N         = x_shape[0];
    int C         = x_shape[1];
    int H         = x_shape[2];
    int W         = x_shape[3];
    int GY_H      = gy_shape[2];
    int GY_W      = gy_shape[3];

    auto result      = std::make_unique<OriginMat>(x_shape, gy.dtype(), gy.device());
    void *gx_data    = result->storage()->data();
    size_t data_size = x_shape.elements() * element_size(gy.dtype());
    CUDA_CHECK(cudaMemset(gx_data, 0, data_size));

    const void *gy_data = gy.storage()->data();

    const size_t threads_per_block = 256;
    const size_t num_elements      = gy_shape.elements();
    const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

    (void)mode;
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        upsample_backward_kernel<T><<<num_blocks, threads_per_block>>>(
            static_cast<const T *>(gy_data), static_cast<T *>(gx_data), N, C, H, W, GY_H, GY_W, scale_h, scale_w);
    });

    return result;
}

}  // namespace cuda
}  // namespace origin
