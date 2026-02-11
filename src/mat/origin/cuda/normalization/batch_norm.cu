#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
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

// ==================== CUDA Kernels ====================

/**
 * @brief 计算每个通道的均值和方差（BatchNorm1d: (N, C)）
 */
template <typename T>
__global__ void compute_mean_var_1d_kernel(const T *__restrict__ x,
                                           T *__restrict__ mean,
                                           T *__restrict__ var,
                                           size_t N,
                                           size_t C)
{
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C)
    {
        T sum        = T(0);
        T sum_sq     = T(0);
        size_t count = 0;

        for (size_t n = 0; n < N; ++n)
        {
            size_t idx = n * C + c;
            T val      = x[idx];
            sum += val;
            sum_sq += val * val;
            count++;
        }

        if (count > 0)
        {
            T mean_val = sum / static_cast<T>(count);
            mean[c]    = mean_val;
            var[c]     = (sum_sq / static_cast<T>(count)) - mean_val * mean_val;
        }
    }
}

/**
 * @brief 计算每个通道的均值和方差（BatchNorm2d: (N, C, H, W)）
 */
template <typename T>
__global__ void compute_mean_var_2d_kernel(const T *__restrict__ x,
                                           T *__restrict__ mean,
                                           T *__restrict__ var,
                                           size_t N,
                                           size_t C,
                                           size_t H,
                                           size_t W)
{
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C)
    {
        T sum        = T(0);
        T sum_sq     = T(0);
        size_t count = 0;

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t idx = ((n * C + c) * H + h) * W + w;
                    T val      = x[idx];
                    sum += val;
                    sum_sq += val * val;
                    count++;
                }
            }
        }

        if (count > 0)
        {
            T mean_val = sum / static_cast<T>(count);
            mean[c]    = mean_val;
            var[c]     = (sum_sq / static_cast<T>(count)) - mean_val * mean_val;
        }
    }
}

/**
 * @brief 归一化并应用 gamma 和 beta（BatchNorm1d: (N, C)）
 */
template <typename T>
__global__ void normalize_and_affine_1d_kernel(const T *__restrict__ x,
                                               const T *__restrict__ mean,
                                               const T *__restrict__ var,
                                               const T *__restrict__ gamma,
                                               const T *__restrict__ beta,
                                               T *__restrict__ y,
                                               T *__restrict__ x_norm,
                                               size_t N,
                                               size_t C,
                                               T eps)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C;

    if (idx < total_elements)
    {
        size_t n = idx / C;
        size_t c = idx % C;

        T mean_val = mean[c];
        T var_val  = var[c];
        T std_val;
        if constexpr (std::is_same_v<T, float>)
        {
            std_val = ::sqrtf(var_val + eps);  // float: 使用全局命名空间的 sqrtf（CUDA 内置函数）
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            std_val = ::sqrt(var_val + eps);  // double: 使用全局命名空间的 sqrt（CUDA 内置函数）
        }
        else
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "batch_norm only supports float32 and float64");
        }

        T normalized = (x[idx] - mean_val) / std_val;
        x_norm[idx]  = normalized;
        y[idx]       = gamma[c] * normalized + beta[c];
    }
}

/**
 * @brief 归一化并应用 gamma 和 beta（BatchNorm2d: (N, C, H, W)）
 */
template <typename T>
__global__ void normalize_and_affine_2d_kernel(const T *__restrict__ x,
                                               const T *__restrict__ mean,
                                               const T *__restrict__ var,
                                               const T *__restrict__ gamma,
                                               const T *__restrict__ beta,
                                               T *__restrict__ y,
                                               T *__restrict__ x_norm,
                                               size_t N,
                                               size_t C,
                                               size_t H,
                                               size_t W,
                                               T eps)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        size_t n   = idx / (C * H * W);
        size_t rem = idx % (C * H * W);
        size_t c   = rem / (H * W);
        rem        = rem % (H * W);
        size_t h   = rem / W;
        size_t w   = rem % W;

        T mean_val = mean[c];
        T var_val  = var[c];
        T std_val;
        if constexpr (std::is_same_v<T, float>)
        {
            std_val = ::sqrtf(var_val + eps);  // float: 使用全局命名空间的 sqrtf（CUDA 内置函数）
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            std_val = ::sqrt(var_val + eps);  // double: 使用全局命名空间的 sqrt（CUDA 内置函数）
        }
        else
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "batch_norm only supports float32 and float64");
        }

        T normalized = (x[idx] - mean_val) / std_val;
        x_norm[idx]  = normalized;
        y[idx]       = gamma[c] * normalized + beta[c];
    }
}

/**
 * @brief 反向传播：计算 dgamma 和 dbeta，以及 mean_gy 和 mean_gy_xnorm（BatchNorm1d: (N, C)）
 */
template <typename T>
__global__ void compute_dgamma_dbeta_1d_kernel(const T *__restrict__ gy,
                                               const T *__restrict__ x_norm,
                                               T *__restrict__ dgamma,
                                               T *__restrict__ dbeta,
                                               T *__restrict__ mean_gy,
                                               T *__restrict__ mean_gy_xnorm,
                                               size_t N,
                                               size_t C)
{
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C)
    {
        T sum_gy       = T(0);
        T sum_gy_xnorm = T(0);

        for (size_t n = 0; n < N; ++n)
        {
            size_t idx = n * C + c;
            sum_gy += gy[idx];
            sum_gy_xnorm += gy[idx] * x_norm[idx];
        }

        dgamma[c]        = sum_gy_xnorm;
        dbeta[c]         = sum_gy;
        mean_gy[c]       = sum_gy / static_cast<T>(N);
        mean_gy_xnorm[c] = sum_gy_xnorm / static_cast<T>(N);
    }
}

/**
 * @brief 反向传播：计算 dgamma 和 dbeta，以及 mean_gy 和 mean_gy_xnorm（BatchNorm2d: (N, C, H, W)）
 */
template <typename T>
__global__ void compute_dgamma_dbeta_2d_kernel(const T *__restrict__ gy,
                                               const T *__restrict__ x_norm,
                                               T *__restrict__ dgamma,
                                               T *__restrict__ dbeta,
                                               T *__restrict__ mean_gy,
                                               T *__restrict__ mean_gy_xnorm,
                                               size_t N,
                                               size_t C,
                                               size_t H,
                                               size_t W)
{
    size_t c = blockIdx.x * blockDim.x + threadIdx.x;

    if (c < C)
    {
        T sum_gy       = T(0);
        T sum_gy_xnorm = T(0);
        size_t count   = N * H * W;

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t idx = ((n * C + c) * H + h) * W + w;
                    sum_gy += gy[idx];
                    sum_gy_xnorm += gy[idx] * x_norm[idx];
                }
            }
        }

        dgamma[c]        = sum_gy_xnorm;
        dbeta[c]         = sum_gy;
        mean_gy[c]       = sum_gy / static_cast<T>(count);
        mean_gy_xnorm[c] = sum_gy_xnorm / static_cast<T>(count);
    }
}

/**
 * @brief 反向传播：计算 gx（BatchNorm1d: (N, C)）
 * @param mean_gy 每个通道的 gy 均值（预先计算）
 * @param mean_gy_xnorm 每个通道的 gy * x_norm 均值（预先计算）
 */
template <typename T>
__global__ void compute_gx_1d_kernel(const T *__restrict__ gy,
                                     const T *__restrict__ gamma,
                                     const T *__restrict__ x_norm,
                                     const T *__restrict__ saved_var,
                                     const T *__restrict__ mean_gy,
                                     const T *__restrict__ mean_gy_xnorm,
                                     T *__restrict__ gx,
                                     size_t N,
                                     size_t C,
                                     T eps)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C;

    if (idx < total_elements)
    {
        size_t c = idx % C;
        T std_val;
        if constexpr (std::is_same_v<T, float>)
        {
            std_val = ::sqrtf(saved_var[c] + eps);  // float: 使用全局命名空间的 sqrtf（CUDA 内置函数）
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            std_val = ::sqrt(saved_var[c] + eps);  // double: 使用全局命名空间的 sqrt（CUDA 内置函数）
        }
        else
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "batch_norm only supports float32 and float64");
        }
        gx[idx] = (gamma[c] / std_val) * (gy[idx] - mean_gy[c] - x_norm[idx] * mean_gy_xnorm[c]);
    }
}

/**
 * @brief 反向传播：计算 gx（BatchNorm2d: (N, C, H, W)）
 * @param mean_gy 每个通道的 gy 均值（预先计算）
 * @param mean_gy_xnorm 每个通道的 gy * x_norm 均值（预先计算）
 */
template <typename T>
__global__ void compute_gx_2d_kernel(const T *__restrict__ gy,
                                     const T *__restrict__ gamma,
                                     const T *__restrict__ x_norm,
                                     const T *__restrict__ saved_var,
                                     const T *__restrict__ mean_gy,
                                     const T *__restrict__ mean_gy_xnorm,
                                     T *__restrict__ gx,
                                     size_t N,
                                     size_t C,
                                     size_t H,
                                     size_t W,
                                     T eps)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        size_t rem = idx % (C * H * W);
        size_t c   = rem / (H * W);
        T std_val;
        if constexpr (std::is_same_v<T, float>)
        {
            std_val = ::sqrtf(saved_var[c] + eps);  // float: 使用全局命名空间的 sqrtf（CUDA 内置函数）
        }
        else if constexpr (std::is_same_v<T, double>)
        {
            std_val = ::sqrt(saved_var[c] + eps);  // double: 使用全局命名空间的 sqrt（CUDA 内置函数）
        }
        else
        {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "batch_norm only supports float32 and float64");
        }
        gx[idx] = (gamma[c] / std_val) * (gy[idx] - mean_gy[c] - x_norm[idx] * mean_gy_xnorm[c]);
    }
}

// ==================== batch_norm 前向传播 ====================

BatchNormForwardResult batch_norm_forward(const OriginMat &x,
                                          const OriginMat &gamma,
                                          const OriginMat &beta,
                                          const OriginMat &running_mean,
                                          const OriginMat &running_var,
                                          bool training,
                                          float eps,
                                          int num_dims)
{
    // 输入验证
    auto x_shape = x.shape();
    if (unlikely(x_shape.size() != static_cast<size_t>(num_dims)))
    {
        THROW_INVALID_ARG("batch_norm_forward: x must have {} dimensions, but got shape {}", num_dims,
                          x_shape.to_string());
    }

    if (num_dims != 2 && num_dims != 4)
    {
        THROW_INVALID_ARG("batch_norm_forward: num_dims must be 2 or 4, but got {}", num_dims);
    }

    size_t num_channels = x_shape[1];

    // 验证参数形状
    if (gamma.shape() != Shape({num_channels}) || beta.shape() != Shape({num_channels}) ||
        running_mean.shape() != Shape({num_channels}) || running_var.shape() != Shape({num_channels}))
    {
        THROW_INVALID_ARG("batch_norm_forward: gamma, beta, running_mean, running_var must have shape ({})",
                          num_channels);
    }

    // 验证数据类型：batch_norm 只支持浮点类型（与 PyTorch 一致）
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("batch_norm_forward: input x must be float32 or float64, but got {}",
                          static_cast<int>(x.dtype()));
    }
    if (gamma.dtype() != x.dtype() || beta.dtype() != x.dtype() || running_mean.dtype() != x.dtype() ||
        running_var.dtype() != x.dtype())
    {
        THROW_INVALID_ARG(
            "batch_norm_forward: all inputs (x, gamma, beta, running_mean, running_var) must have the "
            "same floating-point dtype");
    }

    VALIDATE_CUDA_DEVICE(x);

    // 创建输出
    auto y      = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto mean   = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
    auto var    = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
    auto x_norm = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());

    // 获取数据指针
    const void *x_data            = x.storage()->data();
    const void *gamma_data        = gamma.storage()->data();
    const void *beta_data         = beta.storage()->data();
    const void *running_mean_data = running_mean.storage()->data();
    const void *running_var_data  = running_var.storage()->data();

    void *y_data      = y->storage()->data();
    void *mean_data   = mean->storage()->data();
    void *var_data    = var->storage()->data();
    void *x_norm_data = x_norm->storage()->data();

    // 直接调用特定类型的实现（只支持浮点类型）
    if (x.dtype() == DataType::kFloat32)
    {
        const float *x_ptr            = static_cast<const float *>(x_data);
        const float *gamma_ptr        = static_cast<const float *>(gamma_data);
        const float *beta_ptr         = static_cast<const float *>(beta_data);
        const float *running_mean_ptr = static_cast<const float *>(running_mean_data);
        const float *running_var_ptr  = static_cast<const float *>(running_var_data);

        float *y_ptr      = static_cast<float *>(y_data);
        float *mean_ptr   = static_cast<float *>(mean_data);
        float *var_ptr    = static_cast<float *>(var_data);
        float *x_norm_ptr = static_cast<float *>(x_norm_data);

        // 计算均值和方差
        if (training)
        {
            // 训练模式：计算当前 batch 的均值和方差
            int threads_per_block = 256;
            int num_blocks        = (num_channels + threads_per_block - 1) / threads_per_block;

            if (num_dims == 2)  // BatchNorm1d: (N, C)
            {
                size_t N = x_shape[0];
                compute_mean_var_1d_kernel<float>
                    <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, N, num_channels);
            }
            else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
            {
                size_t N = x_shape[0];
                size_t H = x_shape[2];
                size_t W = x_shape[3];
                compute_mean_var_2d_kernel<float>
                    <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, N, num_channels, H, W);
            }
            CUDA_CHECK_ASYNC();
        }
        else
        {
            // 推理模式：使用 running_mean 和 running_var
            cudaMemcpy(mean_ptr, running_mean_ptr, num_channels * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(var_ptr, running_var_ptr, num_channels * sizeof(float), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ASYNC();
        }

        // 归一化并应用 gamma 和 beta
        size_t total_elements = x_shape.elements();
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        if (num_dims == 2)  // BatchNorm1d: (N, C)
        {
            size_t N = x_shape[0];
            normalize_and_affine_1d_kernel<float>
                <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, y_ptr, x_norm_ptr, N,
                                                    num_channels, static_cast<float>(eps));
        }
        else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
        {
            size_t N = x_shape[0];
            size_t H = x_shape[2];
            size_t W = x_shape[3];
            normalize_and_affine_2d_kernel<float>
                <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, y_ptr, x_norm_ptr, N,
                                                    num_channels, H, W, static_cast<float>(eps));
        }
        CUDA_CHECK_ASYNC();
    }
    else if (x.dtype() == DataType::kFloat64)
    {
        const double *x_ptr            = static_cast<const double *>(x_data);
        const double *gamma_ptr        = static_cast<const double *>(gamma_data);
        const double *beta_ptr         = static_cast<const double *>(beta_data);
        const double *running_mean_ptr = static_cast<const double *>(running_mean_data);
        const double *running_var_ptr  = static_cast<const double *>(running_var_data);

        double *y_ptr      = static_cast<double *>(y_data);
        double *mean_ptr   = static_cast<double *>(mean_data);
        double *var_ptr    = static_cast<double *>(var_data);
        double *x_norm_ptr = static_cast<double *>(x_norm_data);

        // 计算均值和方差
        if (training)
        {
            // 训练模式：计算当前 batch 的均值和方差
            int threads_per_block = 256;
            int num_blocks        = (num_channels + threads_per_block - 1) / threads_per_block;

            if (num_dims == 2)  // BatchNorm1d: (N, C)
            {
                size_t N = x_shape[0];
                compute_mean_var_1d_kernel<double>
                    <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, N, num_channels);
            }
            else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
            {
                size_t N = x_shape[0];
                size_t H = x_shape[2];
                size_t W = x_shape[3];
                compute_mean_var_2d_kernel<double>
                    <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, N, num_channels, H, W);
            }
            CUDA_CHECK_ASYNC();
        }
        else
        {
            // 推理模式：使用 running_mean 和 running_var
            cudaMemcpy(mean_ptr, running_mean_ptr, num_channels * sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(var_ptr, running_var_ptr, num_channels * sizeof(double), cudaMemcpyDeviceToDevice);
            CUDA_CHECK_ASYNC();
        }

        // 归一化并应用 gamma 和 beta
        size_t total_elements = x_shape.elements();
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        if (num_dims == 2)  // BatchNorm1d: (N, C)
        {
            size_t N = x_shape[0];
            normalize_and_affine_1d_kernel<double>
                <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, y_ptr, x_norm_ptr, N,
                                                    num_channels, static_cast<double>(eps));
        }
        else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
        {
            size_t N = x_shape[0];
            size_t H = x_shape[2];
            size_t W = x_shape[3];
            normalize_and_affine_2d_kernel<double>
                <<<num_blocks, threads_per_block>>>(x_ptr, mean_ptr, var_ptr, gamma_ptr, beta_ptr, y_ptr, x_norm_ptr, N,
                                                    num_channels, H, W, static_cast<double>(eps));
        }
        CUDA_CHECK_ASYNC();
    }

    BatchNormForwardResult result;
    result.y      = std::move(y);
    result.mean   = std::move(mean);
    result.var    = std::move(var);
    result.x_norm = std::move(x_norm);
    return result;
}

std::unique_ptr<Mat> batch_norm(const OriginMat &x,
                                const OriginMat &gamma,
                                const OriginMat &beta,
                                const OriginMat &running_mean,
                                const OriginMat &running_var,
                                bool training,
                                float eps,
                                float momentum,
                                int num_dims)
{
    auto result = batch_norm_forward(x, gamma, beta, running_mean, running_var, training, eps, num_dims);
    return std::move(result.y);
}

// ==================== batch_norm 反向传播 ====================

std::vector<std::unique_ptr<Mat>> batch_norm_backward(const OriginMat &gy,
                                                      const OriginMat &x,
                                                      const OriginMat &gamma,
                                                      const OriginMat &saved_mean,
                                                      const OriginMat &saved_var,
                                                      const OriginMat &saved_x_norm,
                                                      float eps,
                                                      int num_dims)
{
    // 输入验证
    auto x_shape = x.shape();
    if (unlikely(x_shape.size() != static_cast<size_t>(num_dims)))
    {
        THROW_INVALID_ARG("batch_norm_backward: x must have {} dimensions, but got shape {}", num_dims,
                          x_shape.to_string());
    }

    if (num_dims != 2 && num_dims != 4)
    {
        THROW_INVALID_ARG("batch_norm_backward: num_dims must be 2 or 4, but got {}", num_dims);
    }

    size_t num_channels = x_shape[1];

    // 验证形状
    if (gy.shape() != x_shape || gamma.shape() != Shape({num_channels}) ||
        saved_mean.shape() != Shape({num_channels}) || saved_var.shape() != Shape({num_channels}) ||
        saved_x_norm.shape() != x_shape)
    {
        THROW_INVALID_ARG("batch_norm_backward: shape mismatch");
    }

    // 验证数据类型：batch_norm 只支持浮点类型（与 PyTorch 一致）
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("batch_norm_backward: input x must be float32 or float64, but got {}",
                          static_cast<int>(x.dtype()));
    }
    if (gy.dtype() != x.dtype() || gamma.dtype() != x.dtype() || saved_mean.dtype() != x.dtype() ||
        saved_var.dtype() != x.dtype() || saved_x_norm.dtype() != x.dtype())
    {
        THROW_INVALID_ARG(
            "batch_norm_backward: all inputs (gy, x, gamma, saved_mean, saved_var, saved_x_norm) must "
            "have the same floating-point dtype");
    }

    VALIDATE_CUDA_DEVICE(x);

    // 创建输出
    auto gx     = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto dgamma = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
    auto dbeta  = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());

    // 获取数据指针
    const void *gy_data           = gy.storage()->data();
    const void *gamma_data        = gamma.storage()->data();
    const void *saved_mean_data   = saved_mean.storage()->data();
    const void *saved_var_data    = saved_var.storage()->data();
    const void *saved_x_norm_data = saved_x_norm.storage()->data();

    void *gx_data     = gx->storage()->data();
    void *dgamma_data = dgamma->storage()->data();
    void *dbeta_data  = dbeta->storage()->data();

    // 初始化梯度为0
    cudaMemset(gx_data, 0, x_shape.elements() * element_size(x.dtype()));
    cudaMemset(dgamma_data, 0, num_channels * element_size(x.dtype()));
    cudaMemset(dbeta_data, 0, num_channels * element_size(x.dtype()));
    CUDA_CHECK_ASYNC();

    // 直接调用特定类型的实现（只支持浮点类型）
    if (x.dtype() == DataType::kFloat32)
    {
        const float *gy_ptr           = static_cast<const float *>(gy_data);
        const float *gamma_ptr        = static_cast<const float *>(gamma_data);
        const float *saved_var_ptr    = static_cast<const float *>(saved_var_data);
        const float *saved_x_norm_ptr = static_cast<const float *>(saved_x_norm_data);

        float *gx_ptr     = static_cast<float *>(gx_data);
        float *dgamma_ptr = static_cast<float *>(dgamma_data);
        float *dbeta_ptr  = static_cast<float *>(dbeta_data);

        int threads_per_block = 256;
        int num_blocks        = (num_channels + threads_per_block - 1) / threads_per_block;

        // 计算 dgamma 和 dbeta（同时计算 mean_gy 和 mean_gy_xnorm）
        // 创建临时缓冲区存储 mean_gy 和 mean_gy_xnorm
        auto mean_gy             = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
        auto mean_gy_xnorm       = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
        float *mean_gy_ptr       = mean_gy->data_ptr<float>();
        float *mean_gy_xnorm_ptr = mean_gy_xnorm->data_ptr<float>();

        if (num_dims == 2)  // BatchNorm1d: (N, C)
        {
            size_t N = x_shape[0];
            compute_dgamma_dbeta_1d_kernel<float><<<num_blocks, threads_per_block>>>(
                gy_ptr, saved_x_norm_ptr, dgamma_ptr, dbeta_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, N, num_channels);
        }
        else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
        {
            size_t N = x_shape[0];
            size_t H = x_shape[2];
            size_t W = x_shape[3];
            compute_dgamma_dbeta_2d_kernel<float><<<num_blocks, threads_per_block>>>(
                gy_ptr, saved_x_norm_ptr, dgamma_ptr, dbeta_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, N, num_channels, H, W);
        }
        CUDA_CHECK_ASYNC();

        // 计算 gx
        size_t total_elements = x_shape.elements();
        num_blocks            = (total_elements + threads_per_block - 1) / threads_per_block;

        if (num_dims == 2)  // BatchNorm1d: (N, C)
        {
            size_t N = x_shape[0];
            compute_gx_1d_kernel<float><<<num_blocks, threads_per_block>>>(
                gy_ptr, gamma_ptr, saved_x_norm_ptr, saved_var_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, gx_ptr, N,
                num_channels, static_cast<float>(eps));
        }
        else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
        {
            size_t N = x_shape[0];
            size_t H = x_shape[2];
            size_t W = x_shape[3];
            compute_gx_2d_kernel<float><<<num_blocks, threads_per_block>>>(
                gy_ptr, gamma_ptr, saved_x_norm_ptr, saved_var_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, gx_ptr, N,
                num_channels, H, W, static_cast<float>(eps));
        }
        CUDA_CHECK_ASYNC();
    }
    else if (x.dtype() == DataType::kFloat64)
    {
        const double *gy_ptr           = static_cast<const double *>(gy_data);
        const double *gamma_ptr        = static_cast<const double *>(gamma_data);
        const double *saved_var_ptr    = static_cast<const double *>(saved_var_data);
        const double *saved_x_norm_ptr = static_cast<const double *>(saved_x_norm_data);

        double *gx_ptr     = static_cast<double *>(gx_data);
        double *dgamma_ptr = static_cast<double *>(dgamma_data);
        double *dbeta_ptr  = static_cast<double *>(dbeta_data);

        int threads_per_block = 256;
        int num_blocks        = (num_channels + threads_per_block - 1) / threads_per_block;

        // 计算 dgamma 和 dbeta（同时计算 mean_gy 和 mean_gy_xnorm）
        // 创建临时缓冲区存储 mean_gy 和 mean_gy_xnorm
        auto mean_gy              = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
        auto mean_gy_xnorm        = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
        double *mean_gy_ptr       = mean_gy->data_ptr<double>();
        double *mean_gy_xnorm_ptr = mean_gy_xnorm->data_ptr<double>();

        if (num_dims == 2)  // BatchNorm1d: (N, C)
        {
            size_t N = x_shape[0];
            compute_dgamma_dbeta_1d_kernel<double><<<num_blocks, threads_per_block>>>(
                gy_ptr, saved_x_norm_ptr, dgamma_ptr, dbeta_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, N, num_channels);
        }
        else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
        {
            size_t N = x_shape[0];
            size_t H = x_shape[2];
            size_t W = x_shape[3];
            compute_dgamma_dbeta_2d_kernel<double><<<num_blocks, threads_per_block>>>(
                gy_ptr, saved_x_norm_ptr, dgamma_ptr, dbeta_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, N, num_channels, H, W);
        }
        CUDA_CHECK_ASYNC();

        // 计算 gx
        size_t total_elements = x_shape.elements();
        num_blocks            = (total_elements + threads_per_block - 1) / threads_per_block;

        if (num_dims == 2)  // BatchNorm1d: (N, C)
        {
            size_t N = x_shape[0];
            compute_gx_1d_kernel<double><<<num_blocks, threads_per_block>>>(
                gy_ptr, gamma_ptr, saved_x_norm_ptr, saved_var_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, gx_ptr, N,
                num_channels, static_cast<double>(eps));
        }
        else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
        {
            size_t N = x_shape[0];
            size_t H = x_shape[2];
            size_t W = x_shape[3];
            compute_gx_2d_kernel<double><<<num_blocks, threads_per_block>>>(
                gy_ptr, gamma_ptr, saved_x_norm_ptr, saved_var_ptr, mean_gy_ptr, mean_gy_xnorm_ptr, gx_ptr, N,
                num_channels, H, W, static_cast<double>(eps));
        }
        CUDA_CHECK_ASYNC();
    }

    std::vector<std::unique_ptr<Mat>> outputs;
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    outputs.push_back(std::move(dbeta));
    return outputs;
}

}  // namespace cuda
}  // namespace origin
