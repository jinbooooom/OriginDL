#include <cuda_runtime.h>
#include <cmath>
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

// ==================== 辅助函数 ====================

/**
 * @brief 检查 normalized_shape 是否为支持的 2 的幂次（64, 128, 256, 512, 1024）
 */
inline bool is_valid_normalized_shape(size_t normalized_shape)
{
    return normalized_shape == 64 || normalized_shape == 128 || normalized_shape == 256 ||
           normalized_shape == 512 || normalized_shape == 1024;
}

/**
 * @brief 根据 normalized_shape 获取对应的 NUM_THREADS
 */
inline int get_num_threads(size_t normalized_shape)
{
    if (!is_valid_normalized_shape(normalized_shape))
    {
        return -1;  // 无效值
    }
    return static_cast<int>(normalized_shape);
}

// ==================== Kernel 辅助函数 ====================

template <typename T, const int kWarpSize = 32>
__device__ __forceinline__ T warp_reduce(T val)
{
#pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, mask);
    }
    return val;
}

template <typename T, int NUM_THREADS = 256>
__device__ T block_reduce(T x)
{
    int tid      = threadIdx.x;
    int warp_num = NUM_THREADS / 32;
    static __shared__ T smem[32];
    int warp_i = tid / 32;
    int lane_i = tid % 32;

    T val = x;
    val   = warp_reduce<T>(val);
    if (lane_i == 0)
        smem[warp_i] = val;
    __syncthreads();
    val = (lane_i < warp_num) ? smem[lane_i] : static_cast<T>(0);

    // 对 warp 结果再进行一次 warp-level 归约
    if (warp_num > 1)
    {
        val = warp_reduce<T>(val);
    }

    return val;
}

// ==================== CUDA Kernels ====================

/**
 * @brief RMSNorm 前向传播 kernel
 * @tparam T 数据类型（float32 或 float64）
 * @param x 输入数据指针，形状 (..., normalized_shape)
 * @param gamma 缩放参数指针，形状 (normalized_shape,)
 * @param y 输出数据指针
 * @param rms RMS 值输出指针，形状 (outer_size,)
 * @param outer_size 除最后一维外的所有维度乘积
 * @param normalized_shape 最后一维大小（归一化维度） 要求==NUM_THREADS 只支持64开始的2幂次
 * @param eps 数值稳定性参数
 */
template <typename T, int NUM_THREADS = 256>
__global__ void rms_norm_forward_kernel(const T *__restrict__ x,
                                        const T *__restrict__ gamma,
                                        T *__restrict__ y,
                                        T *__restrict__ rms,
                                        size_t outer_size,
                                        size_t normalized_shape,
                                        T eps)
{
    // 一个block处理一个token
    size_t tid      = threadIdx.x;
    size_t group_id = blockIdx.x;

    if (group_id >= outer_size) return;

    size_t offset = group_id * normalized_shape;

    // 计算 sum(x^2)
    T sum_sq = static_cast<T>(0);
    for (size_t i = tid; i < normalized_shape; i += NUM_THREADS)
    {
        T val   = x[offset + i];
        sum_sq += val * val;
    }

    // Block 内归约
    sum_sq = block_reduce<T, NUM_THREADS>(sum_sq);

    __shared__ T shared_rms_val;
    if (tid == 0)
    {
        T mean_sq = sum_sq / static_cast<T>(normalized_shape);
        shared_rms_val = std::sqrt(mean_sq + eps);
        rms[group_id] = shared_rms_val;
    }
    __syncthreads();

    // 归一化
    T inv_rms = T(1) / shared_rms_val;
    for (size_t i = tid; i < normalized_shape; i += NUM_THREADS)
    {
        size_t idx     = offset + i;
        y[idx]         = gamma[i] * x[idx] * inv_rms;
    }
}

/**
 * @brief RMSNorm 反向传播 kernel
 * @tparam T 数据类型
 * @param gy 输出梯度指针
 * @param x 输入数据指针
 * @param gamma 缩放参数指针
 * @param saved_rms 前向传播保存的 RMS 值指针
 * @param gx 输入梯度指针
 * @param dgamma gamma 梯度指针
 * @param outer_size 除最后一维外的所有维度乘积
 * @param normalized_shape 最后一维大小
 * @param eps 数值稳定性参数
 */
template <typename T, int NUM_THREADS = 256>
__global__ void rms_norm_backward_kernel(const T *__restrict__ gy,
                                         const T *__restrict__ x,
                                         const T *__restrict__ gamma,
                                         const T *__restrict__ saved_rms,
                                         T *__restrict__ gx,
                                         T *__restrict__ dgamma,
                                         size_t outer_size,
                                         size_t normalized_shape,
                                         T eps)
{
    size_t group_id = blockIdx.x;
    size_t tid      = threadIdx.x;

    if (group_id >= outer_size) return;

    size_t offset = group_id * normalized_shape;
    T rms_val     = saved_rms[group_id];
    T inv_rms     = T(1) / rms_val;
    T inv_rms_sq  = inv_rms * inv_rms;

    // 计算 sum(gy * x)
    T sum_gy_x = static_cast<T>(0);
    for (size_t i = tid; i < normalized_shape; i += NUM_THREADS)
    {
        size_t idx = offset + i;
        sum_gy_x += gy[idx] * x[idx];
    }

    // Block 内归约
    sum_gy_x = block_reduce<T, NUM_THREADS>(sum_gy_x);

    // 广播 sum_gy_x 到所有线程
    __shared__ T shared_sum_gy_x;
    if (tid == 0)
    {
        shared_sum_gy_x = sum_gy_x;
    }
    __syncthreads();
    sum_gy_x = shared_sum_gy_x;

    // 计算 correction
    T correction = sum_gy_x * inv_rms_sq / static_cast<T>(normalized_shape);

    // 计算梯度
    for (size_t i = tid; i < normalized_shape; i += NUM_THREADS)
    {
        size_t idx     = offset + i;
        T dgamma_val   = gy[idx] * x[idx] * inv_rms;
        atomicAdd(&dgamma[i], dgamma_val);

        gx[idx] = inv_rms * (gy[idx] - correction * x[idx]);
    }
}

// ==================== RMSNorm 前向传播 ====================

RMSNormForwardResult rms_norm_forward(const OriginMat &x, const OriginMat &gamma, float eps)
{
    // 输入验证
    auto x_shape = x.shape();
    if (unlikely(x_shape.size() == 0))
    {
        THROW_INVALID_ARG("rms_norm: x must have at least 1 dimension, but got scalar");
    }

    size_t last_dim = x_shape[x_shape.size() - 1];

    // 验证 gamma 形状
    if (gamma.shape() != Shape({last_dim}))
    {
        THROW_INVALID_ARG("rms_norm: gamma must have shape ({}) matching the last dimension of x", last_dim);
    }

    // 验证数据类型
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("rms_norm: input x must be float32 or float64, but got {}", static_cast<int>(x.dtype()));
    }
    if (gamma.dtype() != x.dtype())
    {
        THROW_INVALID_ARG("rms_norm: gamma must have the same dtype as x");
    }

    VALIDATE_CUDA_DEVICE(x);

    // 验证 normalized_shape 必须是 64, 128, 256, 512, 1024 之一
    if (unlikely(!is_valid_normalized_shape(last_dim)))
    {
        THROW_INVALID_ARG("rms_norm: CUDA backend requires normalized_shape to be 64, 128, 256, 512, or 1024 (powers of 2), but got {}", last_dim);
    }

    // 计算输出形状：除了最后一维外，其他维度的总数
    size_t outer_size = 1;
    for (size_t i = 0; i < x_shape.size() - 1; ++i)
    {
        outer_size *= x_shape[i];
    }

    // 创建输出
    auto y   = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto rms = std::make_unique<OriginMat>(Shape{outer_size}, x.dtype(), x.device());

    // 获取数据指针
    const void *x_data     = x.storage()->data();
    const void *gamma_data = gamma.storage()->data();

    void *y_data   = y->storage()->data();
    void *rms_data = rms->storage()->data();

    // 根据数据类型调用对应的 kernel
    if (x.dtype() == DataType::kFloat32)
    {
        const float *x_ptr     = static_cast<const float *>(x_data);
        const float *gamma_ptr = static_cast<const float *>(gamma_data);
        float *y_ptr            = static_cast<float *>(y_data);
        float *rms_ptr          = static_cast<float *>(rms_data);

        // 获取对应的 NUM_THREADS
        int num_threads = get_num_threads(last_dim);

        // 启动 kernel
        switch (num_threads)
        {
            case 64:
                rms_norm_forward_kernel<float, 64>
                    <<<outer_size, 64>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                         static_cast<float>(eps));
                break;
            case 128:
                rms_norm_forward_kernel<float, 128>
                    <<<outer_size, 128>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                          static_cast<float>(eps));
                break;
            case 256:
                rms_norm_forward_kernel<float, 256>
                    <<<outer_size, 256>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                          static_cast<float>(eps));
                break;
            case 512:
                rms_norm_forward_kernel<float, 512>
                    <<<outer_size, 512>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                          static_cast<float>(eps));
                break;
            case 1024:
                rms_norm_forward_kernel<float, 1024>
                    <<<outer_size, 1024>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                           static_cast<float>(eps));
                break;
            default:
                THROW_INVALID_ARG("rms_norm: invalid normalized_shape {}", last_dim);
        }

        CUDA_CHECK_ASYNC();
    }
    else if (x.dtype() == DataType::kFloat64)
    {
        const double *x_ptr     = static_cast<const double *>(x_data);
        const double *gamma_ptr = static_cast<const double *>(gamma_data);
        double *y_ptr            = static_cast<double *>(y_data);
        double *rms_ptr          = static_cast<double *>(rms_data);

        // 获取对应的 NUM_THREADS
        int num_threads = get_num_threads(last_dim);

        // 启动 kernel
        switch (num_threads)
        {
            case 64:
                rms_norm_forward_kernel<double, 64>
                    <<<outer_size, 64>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                         static_cast<double>(eps));
                break;
            case 128:
                rms_norm_forward_kernel<double, 128>
                    <<<outer_size, 128>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                          static_cast<double>(eps));
                break;
            case 256:
                rms_norm_forward_kernel<double, 256>
                    <<<outer_size, 256>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                          static_cast<double>(eps));
                break;
            case 512:
                rms_norm_forward_kernel<double, 512>
                    <<<outer_size, 512>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                          static_cast<double>(eps));
                break;
            case 1024:
                rms_norm_forward_kernel<double, 1024>
                    <<<outer_size, 1024>>>(x_ptr, gamma_ptr, y_ptr, rms_ptr, outer_size, last_dim,
                                           static_cast<double>(eps));
                break;
            default:
                THROW_INVALID_ARG("rms_norm: invalid normalized_shape {}", last_dim);
        }

        CUDA_CHECK_ASYNC();
    }

    RMSNormForwardResult result;
    result.y   = std::move(y);
    result.rms = std::move(rms);
    return result;
}

std::unique_ptr<Mat> rms_norm(const OriginMat &x, const OriginMat &gamma, float eps)
{
    auto result = rms_norm_forward(x, gamma, eps);
    return std::move(result.y);
}

// ==================== RMSNorm 反向传播 ====================

std::vector<std::unique_ptr<Mat>> rms_norm_backward(const OriginMat &gy,
                                                    const OriginMat &x,
                                                    const OriginMat &gamma,
                                                    const OriginMat &saved_rms,
                                                    float eps)
{
    // 输入验证
    auto x_shape = x.shape();
    if (unlikely(x_shape.size() == 0))
    {
        THROW_INVALID_ARG("rms_norm_backward: x must have at least 1 dimension, but got scalar");
    }

    size_t last_dim = x_shape[x_shape.size() - 1];

    // 验证形状
    size_t outer_size = 1;
    for (size_t i = 0; i < x_shape.size() - 1; ++i)
    {
        outer_size *= x_shape[i];
    }

    if (gy.shape() != x_shape || gamma.shape() != Shape({last_dim}) || saved_rms.shape() != Shape({outer_size}))
    {
        THROW_INVALID_ARG("rms_norm_backward: shape mismatch");
    }

    // 验证数据类型
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("rms_norm_backward: input x must be float32 or float64, but got {}",
                          static_cast<int>(x.dtype()));
    }
    if (gy.dtype() != x.dtype() || gamma.dtype() != x.dtype() || saved_rms.dtype() != x.dtype())
    {
        THROW_INVALID_ARG(
            "rms_norm_backward: all inputs (gy, x, gamma, saved_rms) must have the same floating-point dtype");
    }

    VALIDATE_CUDA_DEVICE(x);

    // 验证 normalized_shape 必须是 64, 128, 256, 512, 1024 之一
    if (unlikely(!is_valid_normalized_shape(last_dim)))
    {
        THROW_INVALID_ARG("rms_norm_backward: CUDA backend requires normalized_shape to be 64, 128, 256, 512, or 1024 (powers of 2), but got {}", last_dim);
    }

    // 创建输出
    auto gx     = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto dgamma = std::make_unique<OriginMat>(Shape({last_dim}), x.dtype(), x.device());

    // 获取数据指针
    const void *gy_data        = gy.storage()->data();
    const void *x_data         = x.storage()->data();
    const void *gamma_data     = gamma.storage()->data();
    const void *saved_rms_data = saved_rms.storage()->data();

    void *gx_data     = gx->storage()->data();
    void *dgamma_data = dgamma->storage()->data();

    // 根据数据类型调用对应的 kernel
    if (x.dtype() == DataType::kFloat32)
    {
        const float *gy_ptr        = static_cast<const float *>(gy_data);
        const float *x_ptr         = static_cast<const float *>(x_data);
        const float *gamma_ptr     = static_cast<const float *>(gamma_data);
        const float *saved_rms_ptr = static_cast<const float *>(saved_rms_data);
        float *gx_ptr               = static_cast<float *>(gx_data);
        float *dgamma_ptr           = static_cast<float *>(dgamma_data);

        // 初始化 dgamma 为 0
        cudaMemsetAsync(dgamma_ptr, 0, last_dim * sizeof(float));

        // 获取对应的 NUM_THREADS
        int num_threads = get_num_threads(last_dim);

        // 启动 kernel
        switch (num_threads)
        {
            case 64:
                rms_norm_backward_kernel<float, 64>
                    <<<outer_size, 64>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                          dgamma_ptr, outer_size, last_dim, static_cast<float>(eps));
                break;
            case 128:
                rms_norm_backward_kernel<float, 128>
                    <<<outer_size, 128>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                           dgamma_ptr, outer_size, last_dim, static_cast<float>(eps));
                break;
            case 256:
                rms_norm_backward_kernel<float, 256>
                    <<<outer_size, 256>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                           dgamma_ptr, outer_size, last_dim, static_cast<float>(eps));
                break;
            case 512:
                rms_norm_backward_kernel<float, 512>
                    <<<outer_size, 512>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                           dgamma_ptr, outer_size, last_dim, static_cast<float>(eps));
                break;
            case 1024:
                rms_norm_backward_kernel<float, 1024>
                    <<<outer_size, 1024>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                            dgamma_ptr, outer_size, last_dim, static_cast<float>(eps));
                break;
            default:
                THROW_INVALID_ARG("rms_norm_backward: invalid normalized_shape {}", last_dim);
        }

        CUDA_CHECK_ASYNC();
    }
    else if (x.dtype() == DataType::kFloat64)
    {
        const double *gy_ptr        = static_cast<const double *>(gy_data);
        const double *x_ptr         = static_cast<const double *>(x_data);
        const double *gamma_ptr     = static_cast<const double *>(gamma_data);
        const double *saved_rms_ptr = static_cast<const double *>(saved_rms_data);
        double *gx_ptr               = static_cast<double *>(gx_data);
        double *dgamma_ptr           = static_cast<double *>(dgamma_data);

        // 初始化 dgamma 为 0
        cudaMemsetAsync(dgamma_ptr, 0, last_dim * sizeof(double));

        // 获取对应的 NUM_THREADS
        int num_threads = get_num_threads(last_dim);

        // 启动 kernel
        switch (num_threads)
        {
            case 64:
                rms_norm_backward_kernel<double, 64>
                    <<<outer_size, 64>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                          dgamma_ptr, outer_size, last_dim, static_cast<double>(eps));
                break;
            case 128:
                rms_norm_backward_kernel<double, 128>
                    <<<outer_size, 128>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                           dgamma_ptr, outer_size, last_dim, static_cast<double>(eps));
                break;
            case 256:
                rms_norm_backward_kernel<double, 256>
                    <<<outer_size, 256>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                           dgamma_ptr, outer_size, last_dim, static_cast<double>(eps));
                break;
            case 512:
                rms_norm_backward_kernel<double, 512>
                    <<<outer_size, 512>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                           dgamma_ptr, outer_size, last_dim, static_cast<double>(eps));
                break;
            case 1024:
                rms_norm_backward_kernel<double, 1024>
                    <<<outer_size, 1024>>>(gy_ptr, x_ptr, gamma_ptr, saved_rms_ptr, gx_ptr,
                                            dgamma_ptr, outer_size, last_dim, static_cast<double>(eps));
                break;
            default:
                THROW_INVALID_ARG("rms_norm_backward: invalid normalized_shape {}", last_dim);
        }

        CUDA_CHECK_ASYNC();
    }

    std::vector<std::unique_ptr<Mat>> outputs;
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    return outputs;
}

}  // namespace cuda
}  // namespace origin
