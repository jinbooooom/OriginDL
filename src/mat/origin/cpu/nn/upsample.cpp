#include <cmath>
#include <memory>
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
 * @brief CPU upsample：上采样操作（最近邻）
 * @param x 输入张量 (N, C, H, W)
 * @param output_shape 输出形状 (N, C, OH, OW)
 * @param scale_h 高度缩放因子
 * @param scale_w 宽度缩放因子
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> upsample(const OriginMat &x, const Shape &output_shape, int scale_h, int scale_w)
{
    if (unlikely(x.shape().size() != 4))
    {
        THROW_INVALID_ARG("Upsample: input must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }

    if (unlikely(output_shape.size() != 4))
    {
        THROW_INVALID_ARG("Upsample: output_shape must be 4D (N, C, OH, OW), but got shape {}",
                          output_shape.to_string());
    }

    auto x_shape = x.shape();
    int N        = x_shape[0];
    int C        = x_shape[1];
    int H        = x_shape[2];
    int W        = x_shape[3];
    int OH       = output_shape[2];
    int OW       = output_shape[3];

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(output_shape, x.dtype(), x.device());

    const void *x_data = x.storage()->data();
    void *y_data       = result->storage()->data();

    // 使用类型分发器执行上采样操作
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *x_ptr = static_cast<const T *>(x_data);
        T *y_ptr       = static_cast<T *>(y_data);

        // 最近邻上采样：每个输入像素复制 scale_h * scale_w 次
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
                        int input_idx  = ((n * C + c) * H + ih) * W + iw;
                        int output_idx = ((n * C + c) * OH + oh) * OW + ow;

                        y_ptr[output_idx] = x_ptr[input_idx];
                    }
                }
            }
        }
    });

    return result;
}

/**
 * @brief CPU upsample_backward：上采样反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x_shape 输入形状 (N, C, H, W)
 * @param scale_h 高度缩放因子
 * @param scale_w 宽度缩放因子
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> upsample_backward(const OriginMat &gy, const Shape &x_shape, int scale_h, int scale_w)
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

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(x_shape, gy.dtype(), gy.device());

    const void *gy_data = gy.storage()->data();
    void *gx_data       = result->storage()->data();

    // 使用类型分发器执行反向传播操作
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_ptr = static_cast<const T *>(gy_data);
        T *gx_ptr       = static_cast<T *>(gx_data);

        // 初始化梯度为0
        for (size_t i = 0; i < x_shape.elements(); ++i)
        {
            gx_ptr[i] = T(0);
        }

        // 下采样梯度：对每个输入像素，累加所有对应的输出梯度
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

                        gx_ptr[gx_idx] += gy_ptr[gy_idx];
                    }
                }
            }
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
