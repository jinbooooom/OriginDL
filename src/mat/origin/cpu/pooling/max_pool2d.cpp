#include <algorithm>
#include <cstring>
#include <vector>
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/conv_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

// ==================== max_pool2d 实现 ====================

/**
 * @brief 计算最大值并返回索引
 * @tparam T 数据类型
 * @param data 数据指针
 * @param size 数据大小
 * @return 最大值和索引的pair
 */
template <typename T>
std::pair<T, size_t> max_with_index(T *data, size_t size)
{
    T max_val      = data[0];
    size_t max_idx = 0;
    for (size_t i = 1; i < size; ++i)
    {
        if (data[i] > max_val)
        {
            max_val = data[i];
            max_idx = i;
        }
    }
    return std::make_pair(max_val, max_idx);
}

std::unique_ptr<Mat> max_pool2d(const OriginMat &x,
                                std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad,
                                std::vector<size_t> &indices)
{
    // 输入验证：确保输入是4D张量 (N, C, H, W)
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("max_pool2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }

    size_t N = x.shape()[0];
    size_t C = x.shape()[1];
    size_t H = x.shape()[2];
    size_t W = x.shape()[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;
    int SH = stride.first;
    int SW = stride.second;
    int PH = pad.first;
    int PW = pad.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    if (OH <= 0 || OW <= 0)
    {
        THROW_INVALID_ARG(
            "max_pool2d: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
            "stride=({},{}), pad=({},{})",
            OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 1. 使用 im2col 提取窗口（to_matrix=false），得到 (N, C, KH, KW, OH, OW)
    auto col                 = im2col(x, kernel_size, stride, pad, false);
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // 2. 在 (KH, KW) 维度上求最大值，并保存索引
    // col_shape 是 (N, C, KH, KW, OH, OW)
    Shape output_shape{N, C, static_cast<size_t>(OH), static_cast<size_t>(OW)};
    auto result = std::make_unique<OriginMat>(output_shape, x.dtype(), x.device());

    // 清空并准备索引向量
    indices.clear();
    indices.resize(N * C * OH * OW);

    // 使用类型分发器计算最大值和索引
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const T *col_data = col_mat.data_ptr<T>();
        T *result_data    = result->data_ptr<T>();

        // 对于每个 (n, c, oh, ow)，在 (KH, KW) 维度上求最大值
        // col的形状是 (N, C, KH, KW, OH, OW)
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t c = 0; c < C; ++c)
            {
                for (size_t oh = 0; oh < static_cast<size_t>(OH); ++oh)
                {
                    for (size_t ow = 0; ow < static_cast<size_t>(OW); ++ow)
                    {
                        // 计算在col中的起始位置
                        // col索引: (n, c, kh, kw, oh, ow)
                        // 对于固定的(n, c, oh, ow)，遍历所有(kh, kw)求最大值
                        size_t first_col_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW + 0 * KW * OH * OW +
                                               0 * OH * OW + oh * OW + ow;
                        T max_val      = col_data[first_col_idx];
                        size_t max_idx = 0;

                        for (int kh = 0; kh < KH; ++kh)
                        {
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                size_t col_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW +
                                                 static_cast<size_t>(kh) * KW * OH * OW +
                                                 static_cast<size_t>(kw) * OH * OW + oh * OW + ow;
                                T val             = col_data[col_idx];
                                size_t linear_idx = static_cast<size_t>(kh) * KW + static_cast<size_t>(kw);
                                if (val > max_val)
                                {
                                    max_val = val;
                                    max_idx = linear_idx;
                                }
                            }
                        }

                        // 保存结果
                        size_t result_idx       = n * C * OH * OW + c * OH * OW + oh * OW + ow;
                        result_data[result_idx] = max_val;
                        indices[result_idx]     = max_idx;
                    }
                }
            }
        }
    });

    return result;
}

std::unique_ptr<Mat> max_pool2d_backward(const OriginMat &gy,
                                         const OriginMat &x,
                                         std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride,
                                         std::pair<int, int> pad,
                                         const std::vector<size_t> &indices)
{
    // 输入验证：确保 gy 形状为 (N, C, OH, OW)
    if (gy.shape().size() != 4)
    {
        THROW_INVALID_ARG("max_pool2d_backward: gy must be 4D (N, C, OH, OW), but got shape {}",
                          gy.shape().to_string());
    }

    size_t N  = gy.shape()[0];
    size_t C  = gy.shape()[1];
    size_t OH = gy.shape()[2];
    size_t OW = gy.shape()[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;

    // 验证索引大小
    if (indices.size() != N * C * OH * OW)
    {
        THROW_INVALID_ARG("max_pool2d_backward: indices size mismatch. Expected {}, got {}", N * C * OH * OW,
                          indices.size());
    }

    // 1. 创建零张量 gcol，形状 (N, C, KH, KW, OH, OW) 以匹配 col2im 的期望格式
    Shape gcol_shape{N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), OH, OW};
    auto gcol = std::make_unique<OriginMat>(gcol_shape, gy.dtype(), gy.device());

    // 2. 根据索引将 gy 的值放到对应位置
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        const T *gy_data = gy.data_ptr<T>();
        T *gcol_data     = gcol->data_ptr<T>();

        // 初始化gcol为0
        std::memset(gcol_data, 0, N * C * OH * OW * KH * KW * sizeof(T));

        // 对于每个 (n, c, oh, ow)，根据索引将梯度放到对应位置
        for (size_t n = 0; n < N; ++n)
        {
            for (size_t c = 0; c < C; ++c)
            {
                for (size_t oh = 0; oh < OH; ++oh)
                {
                    for (size_t ow = 0; ow < OW; ++ow)
                    {
                        size_t gy_idx = n * C * OH * OW + c * OH * OW + oh * OW + ow;
                        size_t idx    = indices[gy_idx];  // 窗口内的线性索引

                        // 将索引转换为 (kh, kw)
                        int kh = static_cast<int>(idx) / KW;
                        int kw = static_cast<int>(idx) % KW;

                        // 计算gcol中的位置
                        // gcol形状: (N, C, KH, KW, OH, OW)
                        size_t gcol_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW +
                                          static_cast<size_t>(kh) * KW * OH * OW + static_cast<size_t>(kw) * OH * OW +
                                          oh * OW + ow;

                        gcol_data[gcol_idx] = gy_data[gy_idx];
                    }
                }
            }
        }
    });

    const OriginMat &gcol_mat = static_cast<const OriginMat &>(*gcol);

    // 3. 使用 col2im 转换回 (N, C, H, W)
    auto gx = col2im(gcol_mat, x.shape(), kernel_size, stride, pad, false);
    return gx;
}

}  // namespace cpu
}  // namespace origin
