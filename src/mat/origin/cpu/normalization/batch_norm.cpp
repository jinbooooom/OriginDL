#include <algorithm>
#include <cmath>
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
    if (x_shape.size() != static_cast<size_t>(num_dims))
    {
        THROW_INVALID_ARG("batch_norm: x must have {} dimensions, but got shape {}", num_dims, x_shape.to_string());
    }

    if (num_dims != 2 && num_dims != 4)
    {
        THROW_INVALID_ARG("batch_norm: num_dims must be 2 or 4, but got {}", num_dims);
    }

    size_t num_channels = x_shape[1];

    // 验证参数形状
    if (gamma.shape() != Shape({num_channels}) || beta.shape() != Shape({num_channels}) ||
        running_mean.shape() != Shape({num_channels}) || running_var.shape() != Shape({num_channels}))
    {
        THROW_INVALID_ARG("batch_norm: gamma, beta, running_mean, running_var must have shape ({})", num_channels);
    }

    // 验证数据类型：batch_norm 只支持浮点类型（与 PyTorch 一致）
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("batch_norm: input x must be float32 or float64, but got {}", static_cast<int>(x.dtype()));
    }
    if (gamma.dtype() != x.dtype() || beta.dtype() != x.dtype() || running_mean.dtype() != x.dtype() ||
        running_var.dtype() != x.dtype())
    {
        THROW_INVALID_ARG(
            "batch_norm: all inputs (x, gamma, beta, running_mean, running_var) must have the same "
            "floating-point dtype");
    }

    // 获取数据指针
    const void *x_data            = x.storage()->data();
    const void *gamma_data        = gamma.storage()->data();
    const void *beta_data         = beta.storage()->data();
    const void *running_mean_data = running_mean.storage()->data();
    const void *running_var_data  = running_var.storage()->data();

    // 创建输出
    auto y      = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto mean   = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
    auto var    = std::make_unique<OriginMat>(Shape({num_channels}), x.dtype(), x.device());
    auto x_norm = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());

    void *y_data      = y->storage()->data();
    void *mean_data   = mean->storage()->data();
    void *var_data    = var->storage()->data();
    void *x_norm_data = x_norm->storage()->data();

    // 使用类型分发器执行计算
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *x_ptr            = static_cast<const T *>(x_data);
        const T *gamma_ptr        = static_cast<const T *>(gamma_data);
        const T *beta_ptr         = static_cast<const T *>(beta_data);
        const T *running_mean_ptr = static_cast<const T *>(running_mean_data);
        const T *running_var_ptr  = static_cast<const T *>(running_var_data);

        T *y_ptr      = static_cast<T *>(y_data);
        T *mean_ptr   = static_cast<T *>(mean_data);
        T *var_ptr    = static_cast<T *>(var_data);
        T *x_norm_ptr = static_cast<T *>(x_norm_data);

        // 计算均值和方差
        if (training)
        {
            // 训练模式：计算当前 batch 的均值和方差
            for (size_t c = 0; c < num_channels; ++c)
            {
                T sum        = T(0);
                size_t count = 0;

                if (num_dims == 2)  // BatchNorm1d: (N, C)
                {
                    for (size_t n = 0; n < x_shape[0]; ++n)
                    {
                        size_t idx = n * num_channels + c;
                        sum += x_ptr[idx];
                        count++;
                    }
                }
                else if (num_dims == 4)  // BatchNorm2d: (N, C, H, W)
                {
                    for (size_t n = 0; n < x_shape[0]; ++n)
                    {
                        for (size_t h = 0; h < x_shape[2]; ++h)
                        {
                            for (size_t w = 0; w < x_shape[3]; ++w)
                            {
                                size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                                sum += x_ptr[idx];
                                count++;
                            }
                        }
                    }
                }

                if (count > 0)
                {
                    mean_ptr[c] = sum / static_cast<T>(count);
                }

                // 计算方差
                T sum_sq_diff = T(0);
                if (num_dims == 2)
                {
                    for (size_t n = 0; n < x_shape[0]; ++n)
                    {
                        size_t idx = n * num_channels + c;
                        T diff     = x_ptr[idx] - mean_ptr[c];
                        sum_sq_diff += diff * diff;
                    }
                }
                else if (num_dims == 4)
                {
                    for (size_t n = 0; n < x_shape[0]; ++n)
                    {
                        for (size_t h = 0; h < x_shape[2]; ++h)
                        {
                            for (size_t w = 0; w < x_shape[3]; ++w)
                            {
                                size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                                T diff     = x_ptr[idx] - mean_ptr[c];
                                sum_sq_diff += diff * diff;
                            }
                        }
                    }
                }

                if (count > 0)
                {
                    var_ptr[c] = sum_sq_diff / static_cast<T>(count);
                }
            }
        }
        else
        {
            // 推理模式：使用 running_mean 和 running_var
            std::memcpy(mean_ptr, running_mean_ptr, num_channels * sizeof(T));
            std::memcpy(var_ptr, running_var_ptr, num_channels * sizeof(T));
        }

        // 归一化：x_norm = (x - mean) / sqrt(var + eps)
        if (num_dims == 2)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                for (size_t c = 0; c < num_channels; ++c)
                {
                    size_t idx      = n * num_channels + c;
                    T mean_val      = mean_ptr[c];
                    T var_val       = var_ptr[c];
                    T std_val       = std::sqrt(var_val + static_cast<T>(eps));
                    x_norm_ptr[idx] = (x_ptr[idx] - mean_val) / std_val;
                }
            }
        }
        else if (num_dims == 4)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                for (size_t c = 0; c < num_channels; ++c)
                {
                    T mean_val = mean_ptr[c];
                    T var_val  = var_ptr[c];
                    T std_val  = std::sqrt(var_val + static_cast<T>(eps));
                    for (size_t h = 0; h < x_shape[2]; ++h)
                    {
                        for (size_t w = 0; w < x_shape[3]; ++w)
                        {
                            size_t idx      = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                            x_norm_ptr[idx] = (x_ptr[idx] - mean_val) / std_val;
                        }
                    }
                }
            }
        }

        // 应用 gamma 和 beta：y = gamma * x_norm + beta
        if (num_dims == 2)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                for (size_t c = 0; c < num_channels; ++c)
                {
                    size_t idx = n * num_channels + c;
                    y_ptr[idx] = gamma_ptr[c] * x_norm_ptr[idx] + beta_ptr[c];
                }
            }
        }
        else if (num_dims == 4)
        {
            for (size_t n = 0; n < x_shape[0]; ++n)
            {
                for (size_t c = 0; c < num_channels; ++c)
                {
                    T gamma_val = gamma_ptr[c];
                    T beta_val  = beta_ptr[c];
                    for (size_t h = 0; h < x_shape[2]; ++h)
                    {
                        for (size_t w = 0; w < x_shape[3]; ++w)
                        {
                            size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                            y_ptr[idx] = gamma_val * x_norm_ptr[idx] + beta_val;
                        }
                    }
                }
            }
        }
    });

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
    if (x_shape.size() != static_cast<size_t>(num_dims))
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

    // 使用类型分发器执行计算
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *gy_ptr           = static_cast<const T *>(gy_data);
        const T *gamma_ptr        = static_cast<const T *>(gamma_data);
        const T *saved_var_ptr    = static_cast<const T *>(saved_var_data);
        const T *saved_x_norm_ptr = static_cast<const T *>(saved_x_norm_data);

        T *gx_ptr     = static_cast<T *>(gx_data);
        T *dgamma_ptr = static_cast<T *>(dgamma_data);
        T *dbeta_ptr  = static_cast<T *>(dbeta_data);

        // 初始化梯度
        std::fill(gx_ptr, gx_ptr + x_shape.elements(), T(0));
        std::fill(dgamma_ptr, dgamma_ptr + num_channels, T(0));
        std::fill(dbeta_ptr, dbeta_ptr + num_channels, T(0));

        // 计算每个通道的统计量和梯度
        for (size_t c = 0; c < num_channels; ++c)
        {
            T sum_gy       = T(0);
            T sum_gy_xnorm = T(0);
            size_t count   = 0;

            if (num_dims == 2)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    size_t idx = n * num_channels + c;
                    sum_gy += gy_ptr[idx];
                    sum_gy_xnorm += gy_ptr[idx] * saved_x_norm_ptr[idx];
                    count++;
                }
            }
            else if (num_dims == 4)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    for (size_t h = 0; h < x_shape[2]; ++h)
                    {
                        for (size_t w = 0; w < x_shape[3]; ++w)
                        {
                            size_t idx = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                            sum_gy += gy_ptr[idx];
                            sum_gy_xnorm += gy_ptr[idx] * saved_x_norm_ptr[idx];
                            count++;
                        }
                    }
                }
            }

            T mean_gy       = (count > 0) ? sum_gy / static_cast<T>(count) : T(0);
            T mean_gy_xnorm = (count > 0) ? sum_gy_xnorm / static_cast<T>(count) : T(0);
            T std_val       = std::sqrt(saved_var_ptr[c] + static_cast<T>(eps));

            // 计算 dgamma 和 dbeta
            dgamma_ptr[c] = sum_gy_xnorm;
            dbeta_ptr[c]  = sum_gy;

            // 计算 gx
            if (num_dims == 2)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    size_t idx = n * num_channels + c;
                    gx_ptr[idx] =
                        (gamma_ptr[c] / std_val) * (gy_ptr[idx] - mean_gy - saved_x_norm_ptr[idx] * mean_gy_xnorm);
                }
            }
            else if (num_dims == 4)
            {
                for (size_t n = 0; n < x_shape[0]; ++n)
                {
                    for (size_t h = 0; h < x_shape[2]; ++h)
                    {
                        for (size_t w = 0; w < x_shape[3]; ++w)
                        {
                            size_t idx  = ((n * num_channels + c) * x_shape[2] + h) * x_shape[3] + w;
                            gx_ptr[idx] = (gamma_ptr[c] / std_val) *
                                          (gy_ptr[idx] - mean_gy - saved_x_norm_ptr[idx] * mean_gy_xnorm);
                        }
                    }
                }
            }
        }
    });

    std::vector<std::unique_ptr<Mat>> outputs;
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    outputs.push_back(std::move(dbeta));
    return outputs;
}

}  // namespace cpu
}  // namespace origin
