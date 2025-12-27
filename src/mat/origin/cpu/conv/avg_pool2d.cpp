#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include "origin/utils/conv_utils.h"
#include <vector>
#include <algorithm>
#include <cstring>

namespace origin
{
namespace cpu
{

// ==================== avg_pool2d 实现 ====================

std::unique_ptr<Mat> avg_pool2d(const OriginMat &x, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                                std::pair<int, int> pad)
{
    // 输入验证：确保输入是4D张量 (N, C, H, W)
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("avg_pool2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
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
        THROW_INVALID_ARG("avg_pool2d: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
                         "stride=({},{}), pad=({},{})",
                         OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 1. 使用 im2col 提取窗口（to_matrix=false），得到 (N, C, KH, KW, OH, OW)
    auto col = im2col(x, kernel_size, stride, pad, false);
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // 2. 在 (KH, KW) 维度上求平均值
    // col_shape 是 (N, C, KH, KW, OH, OW)
    // 先对维度2（KH）求和，得到 (N, C, KW, OH, OW)
    auto sum_kh = cpu::sum(col_mat, 2);
    const OriginMat &sum_kh_mat = static_cast<const OriginMat &>(*sum_kh);
    // 再对维度2（KW）求和，得到 (N, C, OH, OW)
    auto sum_kw = cpu::sum(sum_kh_mat, 2);
    const OriginMat &sum_kw_mat = static_cast<const OriginMat &>(*sum_kw);

    // 4. 除以 KH * KW 得到平均值
    float divisor = static_cast<float>(KH * KW);
    auto divisor_tensor = OriginMat::from_scalar(Scalar(divisor), Shape{1}, TensorOptions(x.dtype()));
    auto divisor_broadcast = divisor_tensor->broadcast_to(sum_kw_mat.shape());
    const OriginMat &divisor_mat = static_cast<const OriginMat &>(*divisor_broadcast);
    auto result = sum_kw_mat / divisor_mat;

    return result;
}

std::unique_ptr<Mat> avg_pool2d_backward(const OriginMat &gy, const OriginMat &x, std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride, std::pair<int, int> pad)
{
    // 输入验证：确保 gy 形状为 (N, C, OH, OW)
    if (gy.shape().size() != 4)
    {
        THROW_INVALID_ARG("avg_pool2d_backward: gy must be 4D (N, C, OH, OW), but got shape {}", gy.shape().to_string());
    }

    size_t N = gy.shape()[0];
    size_t C = gy.shape()[1];
    size_t OH = gy.shape()[2];
    size_t OW = gy.shape()[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;

    // 1. 将 gy 广播到 (N, C, OH, OW, KH, KW)
    Shape broadcast_shape{N, C, OH, OW, static_cast<size_t>(KH), static_cast<size_t>(KW)};
    auto gy_broadcast = gy.broadcast_to(broadcast_shape);
    const OriginMat &gy_broadcast_mat = static_cast<const OriginMat &>(*gy_broadcast);

    // 2. 除以 KH * KW（平均值的反向）
    float divisor = static_cast<float>(KH * KW);
    auto divisor_tensor = OriginMat::from_scalar(Scalar(divisor), Shape{1}, TensorOptions(gy.dtype()));
    auto divisor_broadcast = divisor_tensor->broadcast_to(broadcast_shape);
    const OriginMat &divisor_mat = static_cast<const OriginMat &>(*divisor_broadcast);
    auto gcol_reshaped = gy_broadcast_mat / divisor_mat;
    const OriginMat &gcol_reshaped_mat = static_cast<const OriginMat &>(*gcol_reshaped);

    // 3. Reshape 为 (N, C, KH, KW, OH, OW) 以匹配 col2im 的期望格式
    // gcol_reshaped 当前是 (N, C, OH, OW, KH, KW)，需要重新排列为 (N, C, KH, KW, OH, OW)
    Shape col_shape{N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), OH, OW};
    
    // 由于 reshape 不能改变元素顺序，我们需要手动重新排列
    // 使用 to_vector 和 from_memory 来重新排列
    auto gcol_data = gcol_reshaped_mat.to_vector<float>();
    std::vector<float> reordered_data(N * C * KH * KW * OH * OW);
    
    // 从 (N, C, OH, OW, KH, KW) 重新排列为 (N, C, KH, KW, OH, OW)
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t oh = 0; oh < OH; ++oh)
            {
                for (size_t ow = 0; ow < OW; ++ow)
                {
                    for (int kh = 0; kh < KH; ++kh)
                    {
                        for (int kw = 0; kw < KW; ++kw)
                        {
                            size_t src_idx = n * C * OH * OW * KH * KW + c * OH * OW * KH * KW + oh * OW * KH * KW +
                                            ow * KH * KW + static_cast<size_t>(kh) * KW + static_cast<size_t>(kw);
                            size_t dst_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW +
                                            static_cast<size_t>(kh) * KW * OH * OW + static_cast<size_t>(kw) * OH * OW +
                                            oh * OW + ow;
                            reordered_data[dst_idx] = gcol_data[src_idx];
                        }
                    }
                }
            }
        }
    }
    
    auto gcol = OriginMat::from_memory(reordered_data.data(), DataType::kFloat32, col_shape,
                                      TensorOptions(gy.dtype()).device(gy.device()));
    const OriginMat &gcol_mat = static_cast<const OriginMat &>(*gcol);

    // 4. 使用 col2im 转换回 (N, C, H, W)
    auto gx = col2im(gcol_mat, x.shape(), kernel_size, stride, pad, false);
    return gx;
}

}  // namespace cpu
}  // namespace origin

