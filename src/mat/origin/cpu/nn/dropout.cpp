#include <cstring>
#include <memory>
#include <random>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU dropout：Dropout 前向传播
 * @param x 输入张量
 * @param p dropout 概率
 * @param training 是否为训练模式
 * @param mask 输出参数：保存 dropout mask
 * @return 输出张量
 */
std::unique_ptr<Mat> dropout(const OriginMat &x, float p, bool training, OriginMat *mask)
{
    if (unlikely(p < 0.0f || p >= 1.0f))
    {
        THROW_INVALID_ARG("Dropout: p must be in [0, 1), but got {}", p);
    }

    auto x_shape = x.shape();
    auto result  = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());

    if (!training)
    {
        // 推理模式：直接返回输入
        const void *x_data = x.storage()->data();
        void *y_data       = result->storage()->data();
        size_t data_size   = x_shape.elements() * element_size(x.dtype());
        std::memcpy(y_data, x_data, data_size);
        return result;
    }

    // 训练模式：生成 dropout mask
    const void *x_data = x.storage()->data();
    void *y_data       = result->storage()->data();

    // 创建 mask（如果需要）
    OriginMat *mask_ptr = nullptr;
    std::unique_ptr<OriginMat> mask_unique;
    if (mask != nullptr)
    {
        mask_ptr = mask;
    }
    else
    {
        mask_unique = std::make_unique<OriginMat>(x_shape, DataType::kFloat32, x.device());
        mask_ptr    = mask_unique.get();
    }

    void *mask_data = mask_ptr->storage()->data();

    // 生成随机 mask 并应用 dropout
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    float scale = 1.0f / (1.0f - p);

    // 使用类型分发器执行 dropout 操作
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *x_ptr     = static_cast<const T *>(x_data);
        T *y_ptr           = static_cast<T *>(y_data);
        float *mask_ptr_f  = static_cast<float *>(mask_data);

        for (size_t i = 0; i < x_shape.elements(); ++i)
        {
            if (dist(gen) < p)
            {
                mask_ptr_f[i] = 0.0f;
                y_ptr[i]      = T(0);
            }
            else
            {
                mask_ptr_f[i] = scale;
                y_ptr[i]      = static_cast<T>(static_cast<float>(x_ptr[i]) * scale);
            }
        }
    });

    return result;
}

/**
 * @brief CPU dropout_backward：Dropout 反向传播
 * @param gy 输出梯度
 * @param mask dropout mask
 * @return 输入梯度
 */
std::unique_ptr<Mat> dropout_backward(const OriginMat &gy, const OriginMat &mask)
{
    if (unlikely(gy.shape() != mask.shape()))
    {
        THROW_INVALID_ARG("Dropout backward: gradient shape {} must match mask shape {}", gy.shape().to_string(),
                          mask.shape().to_string());
    }

    auto gy_shape = gy.shape();
    auto result   = std::make_unique<OriginMat>(gy_shape, gy.dtype(), gy.device());

    const void *gy_data   = gy.storage()->data();
    const void *mask_data = mask.storage()->data();
    void *gx_data         = result->storage()->data();

    // 使用类型分发器执行反向传播操作
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_ptr      = static_cast<const T *>(gy_data);
        const float *mask_ptr = static_cast<const float *>(mask_data);
        T *gx_ptr            = static_cast<T *>(gx_data);

        for (size_t i = 0; i < gy_shape.elements(); ++i)
        {
            gx_ptr[i] = static_cast<T>(static_cast<float>(gy_ptr[i]) * mask_ptr[i]);
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
