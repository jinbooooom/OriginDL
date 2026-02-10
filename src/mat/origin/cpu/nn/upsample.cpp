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

// 本文件仅实现最近邻上采样；mode 参数保留但未使用，双线性未实现。

/**
 * @brief CPU upsample：上采样操作（最近邻）
 */
std::unique_ptr<Mat> upsample(const OriginMat &x,
                              const Shape &output_shape,
                              int scale_h,
                              int scale_w,
                              const std::string &mode)
{
    (void)mode;
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

    auto result   = std::make_unique<OriginMat>(output_shape, x.dtype(), x.device());
    const void *x_data = x.storage()->data();
    void *y_data       = result->storage()->data();

    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *x_ptr = static_cast<const T *>(x_data);
        T *y_ptr       = static_cast<T *>(y_data);
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int oh = 0; oh < OH; ++oh)
                {
                    for (int ow = 0; ow < OW; ++ow)
                    {
                        int ih         = oh / scale_h;
                        int iw         = ow / scale_w;
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
 * @brief CPU upsample_backward：上采样反向传播（最近邻）
 */
std::unique_ptr<Mat> upsample_backward(const OriginMat &gy,
                                       const Shape &x_shape,
                                       int scale_h,
                                       int scale_w,
                                       const std::string &mode)
{
    (void)mode;
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

    auto result   = std::make_unique<OriginMat>(x_shape, gy.dtype(), gy.device());
    const void *gy_data = gy.storage()->data();
    void *gx_data       = result->storage()->data();

    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_ptr = static_cast<const T *>(gy_data);
        T *gx_ptr       = static_cast<T *>(gx_data);
        for (size_t i = 0; i < x_shape.elements(); ++i)
            gx_ptr[i] = T(0);
        for (int n = 0; n < N; ++n)
        {
            for (int c = 0; c < C; ++c)
            {
                for (int gy_h = 0; gy_h < GY_H; ++gy_h)
                {
                    for (int gy_w = 0; gy_w < GY_W; ++gy_w)
                    {
                        int ih     = gy_h / scale_h;
                        int iw     = gy_w / scale_w;
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
