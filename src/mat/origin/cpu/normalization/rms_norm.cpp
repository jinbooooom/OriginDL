#include <cmath>
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

// ==================== rms_norm 前向传播 ====================

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

    // 验证数据类型：rms_norm 只支持浮点类型（与 PyTorch 一致）
    if (x.dtype() != DataType::kFloat32 && x.dtype() != DataType::kFloat64)
    {
        THROW_INVALID_ARG("rms_norm: input x must be float32 or float64, but got {}", static_cast<int>(x.dtype()));
    }
    if (gamma.dtype() != x.dtype())
    {
        THROW_INVALID_ARG("rms_norm: gamma must have the same dtype as x");
    }

    // 获取数据指针
    const void *x_data     = x.storage()->data();
    const void *gamma_data = gamma.storage()->data();

    // 计算输出形状：除了最后一维外，其他维度的总数
    size_t outer_size = 1;
    for (size_t i = 0; i < x_shape.size() - 1; ++i)
    {
        outer_size *= x_shape[i];
    }

    // 创建输出
    auto y   = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());
    auto rms = std::make_unique<OriginMat>(Shape{outer_size}, x.dtype(), x.device());

    void *y_data   = y->storage()->data();
    void *rms_data = rms->storage()->data();

    // 使用类型分发器执行计算
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *x_ptr     = static_cast<const T *>(x_data);
        const T *gamma_ptr = static_cast<const T *>(gamma_data);

        T *y_ptr   = static_cast<T *>(y_data);
        T *rms_ptr = static_cast<T *>(rms_data);

        T eps_val = static_cast<T>(eps);

        // 对每个归一化组（最后一维前面的所有维度）进行计算
        for (size_t i = 0; i < outer_size; ++i)
        {
            // 计算 RMS: sqrt(mean(x^2) + eps)
            T sum_sq = T(0);
            for (size_t j = 0; j < last_dim; ++j)
            {
                size_t idx = i * last_dim + j;
                T val      = x_ptr[idx];
                sum_sq += val * val;
            }

            T mean_sq  = sum_sq / static_cast<T>(last_dim);
            T rms_val  = std::sqrt(mean_sq + eps_val);
            rms_ptr[i] = rms_val;

            // 归一化并应用 gamma：y = gamma * x / rms
            for (size_t j = 0; j < last_dim; ++j)
            {
                size_t idx = i * last_dim + j;
                y_ptr[idx] = gamma_ptr[j] * x_ptr[idx] / rms_val;
            }
        }
    });

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

// ==================== rms_norm 反向传播 ====================

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

    // 验证数据类型：rms_norm 只支持浮点类型（与 PyTorch 一致）
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

    // 使用类型分发器执行计算
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *gy_ptr        = static_cast<const T *>(gy_data);
        const T *x_ptr         = static_cast<const T *>(x_data);
        const T *gamma_ptr     = static_cast<const T *>(gamma_data);
        const T *saved_rms_ptr = static_cast<const T *>(saved_rms_data);

        T *gx_ptr     = static_cast<T *>(gx_data);
        T *dgamma_ptr = static_cast<T *>(dgamma_data);

        T eps_val = static_cast<T>(eps);

        // 初始化 dgamma
        std::fill(dgamma_ptr, dgamma_ptr + last_dim, T(0));

        // 对每个归一化组进行反向计算
        for (size_t i = 0; i < outer_size; ++i)
        {
            T rms_val = saved_rms_ptr[i];

            // 计算 dgamma: sum(gy * x / rms)
            for (size_t j = 0; j < last_dim; ++j)
            {
                size_t idx = i * last_dim + j;
                dgamma_ptr[j] += gy_ptr[idx] * x_ptr[idx] / rms_val;
            }

            // 计算 sum_gy_x: sum(gy * x)
            // 计算 sum_x_sq: sum(x^2)
            T sum_gy_x = T(0);
            T sum_x_sq = T(0);
            for (size_t j = 0; j < last_dim; ++j)
            {
                size_t idx = i * last_dim + j;
                sum_gy_x += gy_ptr[idx] * x_ptr[idx];
                sum_x_sq += x_ptr[idx] * x_ptr[idx];
            }

            // 计算 gx
            // gx = (1 / rms) * (gy - (1 / (rms^2 * last_dim)) * sum_gy_x * x)
            T rms_sq     = rms_val * rms_val;
            T scale      = T(1) / rms_val;
            T correction = sum_gy_x / (rms_sq * static_cast<T>(last_dim));

            for (size_t j = 0; j < last_dim; ++j)
            {
                size_t idx  = i * last_dim + j;
                gx_ptr[idx] = scale * (gy_ptr[idx] - correction * x_ptr[idx]);
            }
        }
    });

    std::vector<std::unique_ptr<Mat>> outputs;
    outputs.push_back(std::move(gx));
    outputs.push_back(std::move(dgamma));
    return outputs;
}

}  // namespace cpu
}  // namespace origin
