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

// ==================== 内部辅助函数：im2col 和 col2im ====================
// 这些函数是 conv2d 的内部实现细节，不对外暴露

namespace
{

/**
 * @brief im2col：将图像转换为列矩阵（内部实现）
 */
std::unique_ptr<Mat> im2col_impl(const OriginMat &img,
                                 std::pair<int, int> kernel_size,
                                 std::pair<int, int> stride,
                                 std::pair<int, int> pad,
                                 bool to_matrix)
{
    auto img_shape = img.shape();

    // 检查输入形状：必须是 (N, C, H, W)
    if (img_shape.size() != 4)
    {
        THROW_INVALID_ARG("im2col: input must be 4D (N, C, H, W), but got shape {}", img_shape.to_string());
    }

    size_t N = img_shape[0];
    size_t C = img_shape[1];
    size_t H = img_shape[2];
    size_t W = img_shape[3];

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
            "im2col: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
            "stride=({},{}), pad=({},{})",
            OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 获取输入数据指针（使用类型分发器支持多种类型）
    std::vector<float> img_data_vec;
    const float *img_data = nullptr;

    if (img.dtype() == DataType::kFloat32)
    {
        img_data = img.data_ptr<float>();
    }
    else
    {
        // 对于非 float32 类型，先转换为 float32 向量
        img_data_vec = img.to_vector<float>();
        img_data     = img_data_vec.data();
    }

    // 创建填充后的图像
    size_t padded_H = H + 2 * PH + SH - 1;
    size_t padded_W = W + 2 * PW + SW - 1;
    std::vector<float> padded_img(N * C * padded_H * padded_W, 0.0f);

    // 填充图像
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t src_idx = n * C * H * W + c * H * W + h * W + w;
                    size_t dst_h   = h + PH;
                    size_t dst_w   = w + PW;
                    size_t dst_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + dst_h * padded_W + dst_w;
                    padded_img[dst_idx] = img_data[src_idx];
                }
            }
        }
    }

    if (to_matrix)
    {
        // 形状: (N*OH*OW, C*KH*KW)
        size_t out_rows = N * OH * OW;
        size_t out_cols = C * KH * KW;
        std::vector<float> col_data(out_rows * out_cols, 0.0f);

        for (size_t n = 0; n < N; ++n)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    size_t row_idx = n * OH * OW + oh * OW + ow;

                    for (size_t c = 0; c < C; ++c)
                    {
                        for (int kh = 0; kh < KH; ++kh)
                        {
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                int h_idx = kh + SH * oh;
                                int w_idx = kw + SW * ow;

                                size_t col_idx = c * KH * KW + kh * KW + kw;
                                size_t out_idx = row_idx * out_cols + col_idx;

                                if (h_idx < static_cast<int>(padded_H) && w_idx < static_cast<int>(padded_W))
                                {
                                    size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W +
                                                     h_idx * padded_W + w_idx;
                                    col_data[out_idx] = padded_img[img_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Shape out_shape{out_rows, out_cols};
        auto result = std::make_unique<OriginMat>(out_shape, img.dtype(), img.device());
        // 复制数据
        float *result_data = result->data_ptr<float>();
        std::memcpy(result_data, col_data.data(), col_data.size() * sizeof(float));
        return result;
    }
    else
    {
        // 形状: (N, C, KH, KW, OH, OW)
        std::vector<float> col_data(N * C * KH * KW * OH * OW, 0.0f);

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t c = 0; c < C; ++c)
            {
                for (int kh = 0; kh < KH; ++kh)
                {
                    for (int kw = 0; kw < KW; ++kw)
                    {
                        for (int oh = 0; oh < OH; ++oh)
                        {
                            for (int ow = 0; ow < OW; ++ow)
                            {
                                int h_idx = kh + SH * oh;
                                int w_idx = kw + SW * ow;

                                size_t idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW + kh * KW * OH * OW +
                                             kw * OH * OW + oh * OW + ow;

                                if (h_idx < static_cast<int>(padded_H) && w_idx < static_cast<int>(padded_W))
                                {
                                    size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W +
                                                     h_idx * padded_W + w_idx;
                                    col_data[idx] = padded_img[img_idx];
                                }
                            }
                        }
                    }
                }
            }
        }

        Shape out_shape{
            N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), static_cast<size_t>(OH), static_cast<size_t>(OW)};
        auto result = std::make_unique<OriginMat>(out_shape, img.dtype(), img.device());
        // 复制数据
        float *result_data = result->data_ptr<float>();
        std::memcpy(result_data, col_data.data(), col_data.size() * sizeof(float));
        return result;
    }
}

/**
 * @brief col2im：将列矩阵转换回图像形状（内部实现）
 */
std::unique_ptr<Mat> col2im_impl(const OriginMat &col,
                                 const Shape &input_shape,
                                 std::pair<int, int> kernel_size,
                                 std::pair<int, int> stride,
                                 std::pair<int, int> pad,
                                 bool to_matrix)
{
    // 检查输入形状
    if (input_shape.size() != 4)
    {
        THROW_INVALID_ARG("col2im: input_shape must be 4D (N, C, H, W), but got shape {}", input_shape.to_string());
    }

    auto col_shape = col.shape();

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;
    int SH = stride.first;
    int SW = stride.second;
    int PH = pad.first;
    int PW = pad.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    // 获取列矩阵数据
    std::vector<float> col_data_vec;
    const float *col_data = nullptr;

    if (col.dtype() == DataType::kFloat32)
    {
        col_data = col.data_ptr<float>();
    }
    else
    {
        col_data_vec = col.to_vector<float>();
        col_data     = col_data_vec.data();
    }

    // 如果是从矩阵形式转换，先 reshape
    std::vector<float> col_reshaped;
    if (to_matrix)
    {
        // 从 (N*OH*OW, C*KH*KW) reshape 为 (N, C, KH, KW, OH, OW)
        if (col_shape.size() != 2 || col_shape[0] != static_cast<size_t>(N * OH * OW) ||
            col_shape[1] != static_cast<size_t>(C * KH * KW))
        {
            THROW_INVALID_ARG(
                "col2im: invalid col shape {} for input_shape {}, kernel=({},{}), "
                "stride=({},{}), pad=({},{})",
                col_shape.to_string(), input_shape.to_string(), KH, KW, SH, SW, PH, PW);
        }

        col_reshaped.resize(N * C * KH * KW * OH * OW);
        for (size_t n = 0; n < N; ++n)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    size_t row_idx = n * OH * OW + oh * OW + ow;
                    for (size_t c = 0; c < C; ++c)
                    {
                        for (int kh = 0; kh < KH; ++kh)
                        {
                            for (int kw = 0; kw < KW; ++kw)
                            {
                                size_t col_idx = c * KH * KW + kh * KW + kw;
                                size_t src_idx = row_idx * (C * KH * KW) + col_idx;
                                size_t dst_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW + kh * KW * OH * OW +
                                                 kw * OH * OW + oh * OW + ow;
                                col_reshaped[dst_idx] = col_data[src_idx];
                            }
                        }
                    }
                }
            }
        }
    }
    else
    {
        // 已经是 (N, C, KH, KW, OH, OW) 形状
        if (col_shape.size() != 6 || col_shape[0] != N || col_shape[1] != C ||
            col_shape[2] != static_cast<size_t>(KH) || col_shape[3] != static_cast<size_t>(KW) ||
            col_shape[4] != static_cast<size_t>(OH) || col_shape[5] != static_cast<size_t>(OW))
        {
            THROW_INVALID_ARG(
                "col2im: invalid col shape {} for input_shape {}, kernel=({},{}), "
                "stride=({},{}), pad=({},{})",
                col_shape.to_string(), input_shape.to_string(), KH, KW, SH, SW, PH, PW);
        }
        col_reshaped.resize(N * C * KH * KW * OH * OW);
        std::memcpy(col_reshaped.data(), col_data, col_reshaped.size() * sizeof(float));
    }

    // 创建输出图像（带填充）
    size_t padded_H = H + 2 * PH + SH - 1;
    size_t padded_W = W + 2 * PW + SW - 1;
    std::vector<float> img_data(N * C * padded_H * padded_W, 0.0f);

    // 将列矩阵转换回图像（累加重叠区域）
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (int kh = 0; kh < KH; ++kh)
            {
                for (int kw = 0; kw < KW; ++kw)
                {
                    for (int oh = 0; oh < OH; ++oh)
                    {
                        for (int ow = 0; ow < OW; ++ow)
                        {
                            int h_idx = kh + SH * oh;
                            int w_idx = kw + SW * ow;

                            if (h_idx < static_cast<int>(padded_H) && w_idx < static_cast<int>(padded_W))
                            {
                                size_t col_idx = n * C * KH * KW * OH * OW + c * KH * KW * OH * OW + kh * KW * OH * OW +
                                                 kw * OH * OW + oh * OW + ow;
                                size_t img_idx =
                                    n * C * padded_H * padded_W + c * padded_H * padded_W + h_idx * padded_W + w_idx;

                                img_data[img_idx] += col_reshaped[col_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // 移除填充，得到最终输出
    std::vector<float> output_data(N * C * H * W, 0.0f);
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t c = 0; c < C; ++c)
        {
            for (size_t h = 0; h < H; ++h)
            {
                for (size_t w = 0; w < W; ++w)
                {
                    size_t src_h   = h + PH;
                    size_t src_w   = w + PW;
                    size_t src_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + src_h * padded_W + src_w;
                    size_t dst_idx = n * C * H * W + c * H * W + h * W + w;
                    output_data[dst_idx] = img_data[src_idx];
                }
            }
        }
    }

    Shape out_shape{N, C, H, W};
    auto result = std::make_unique<OriginMat>(out_shape, col.dtype(), col.device());
    // 复制数据
    float *result_data = result->data_ptr<float>();
    std::memcpy(result_data, output_data.data(), output_data.size() * sizeof(float));
    return result;
}

}  // anonymous namespace

// ==================== 对外接口（供 OriginMat::im2col/col2im 使用）====================

std::unique_ptr<Mat> im2col(const OriginMat &img,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix)
{
    return im2col_impl(img, kernel_size, stride, pad, to_matrix);
}

std::unique_ptr<Mat> col2im(const OriginMat &col,
                            const Shape &input_shape,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix)
{
    return col2im_impl(col, input_shape, kernel_size, stride, pad, to_matrix);
}

// ==================== conv2d 实现 ====================

std::unique_ptr<Mat> conv2d(const OriginMat &x,
                            const OriginMat &W,
                            const OriginMat *b,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad)
{
    // 输入验证
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }
    if (W.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d: W must be 4D (OC, C, KH, KW), but got shape {}", W.shape().to_string());
    }

    size_t N    = x.shape()[0];
    size_t C    = x.shape()[1];
    size_t H    = x.shape()[2];
    size_t W_in = x.shape()[3];

    size_t OC   = W.shape()[0];
    size_t C_in = W.shape()[1];
    size_t KH   = W.shape()[2];
    size_t KW   = W.shape()[3];

    // 检查通道数是否匹配
    if (C != C_in)
    {
        THROW_INVALID_ARG("conv2d: channel mismatch - x has {} channels, but W expects {} channels", C, C_in);
    }

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), static_cast<int>(KH), stride.first, pad.first);
    int OW = get_conv_outsize(static_cast<int>(W_in), static_cast<int>(KW), stride.second, pad.second);

    if (OH <= 0 || OW <= 0)
    {
        THROW_INVALID_ARG(
            "conv2d: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
            "stride=({},{}), pad=({},{})",
            OH, OW, H, W_in, KH, KW, stride.first, stride.second, pad.first, pad.second);
    }

    // 检查偏置
    if (b != nullptr)
    {
        if (b->shape().size() != 1 || b->shape()[0] != OC)
        {
            THROW_INVALID_ARG("conv2d: b must be 1D with size {}, but got shape {}", OC, b->shape().to_string());
        }
    }

    // 创建输出 Mat，形状为 (N, OC, OH, OW)
    Shape out_shape{N, OC, static_cast<size_t>(OH), static_cast<size_t>(OW)};
    auto result = std::make_unique<OriginMat>(out_shape, x.dtype(), x.device());

    // 使用 im2col + matmul 实现卷积
    // 1. im2col: (N, C, H, W) -> (N*OH*OW, C*KH*KW)
    auto col = im2col_impl(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);

    // 2. 将卷积核 reshape 为 (OC, C*KH*KW) 并转置为 (C*KH*KW, OC)
    auto W_reshaped          = W.reshape(Shape{OC, C * KH * KW});
    auto W_T                 = W_reshaped->transpose();
    const OriginMat &W_T_mat = static_cast<const OriginMat &>(*W_T);

    // 3. 矩阵乘法: col @ W_T -> (N*OH*OW, OC)
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);
    auto y_flat              = col_mat.matmul(W_T_mat);

    // 4. 添加偏置（如果存在）
    if (b != nullptr)
    {
        // 广播偏置: (OC,) -> (N*OH*OW, OC)
        auto b_broadcast            = b->broadcast_to(Shape{N * static_cast<size_t>(OH) * static_cast<size_t>(OW), OC});
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        const OriginMat &b_broadcast_mat = static_cast<const OriginMat &>(*b_broadcast);
        y_flat                           = y_flat_mat.operator+(b_broadcast_mat);
    }

    // 5. Reshape 并转置: (N*OH*OW, OC) -> (N, OH, OW, OC) -> (N, OC, OH, OW)
    const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
    auto y_reshaped             = y_flat_mat.reshape(Shape{N, static_cast<size_t>(OH), static_cast<size_t>(OW), OC});

    // 手动转置数据：从 (N, OH, OW, OC) 到 (N, OC, OH, OW)
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const OriginMat &y_reshaped_mat = static_cast<const OriginMat &>(*y_reshaped);
        const T *src_data               = y_reshaped_mat.data_ptr<T>();
        T *dst_data                     = result->data_ptr<T>();

        for (size_t n = 0; n < N; ++n)
        {
            for (size_t oc = 0; oc < OC; ++oc)
            {
                for (int oh = 0; oh < OH; ++oh)
                {
                    for (int ow = 0; ow < OW; ++ow)
                    {
                        size_t src_idx    = n * OH * OW * OC + oh * OW * OC + ow * OC + oc;
                        size_t dst_idx    = n * OC * OH * OW + oc * OH * OW + oh * OW + ow;
                        dst_data[dst_idx] = src_data[src_idx];
                    }
                }
            }
        }
    });

    return result;
}

std::vector<std::unique_ptr<Mat>> conv2d_backward(const OriginMat &gy,
                                                  const OriginMat &x,
                                                  const OriginMat &W,
                                                  const OriginMat *b,
                                                  std::pair<int, int> stride,
                                                  std::pair<int, int> pad)
{
    // 输入验证
    if (gy.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d_backward: gy must be 4D (N, OC, OH, OW), but got shape {}", gy.shape().to_string());
    }
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d_backward: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }
    if (W.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d_backward: W must be 4D (OC, C, KH, KW), but got shape {}", W.shape().to_string());
    }

    size_t N  = x.shape()[0];
    size_t C  = x.shape()[1];
    size_t OC = W.shape()[0];
    size_t KH = W.shape()[2];
    size_t KW = W.shape()[3];

    int OH = static_cast<int>(gy.shape()[2]);
    int OW = static_cast<int>(gy.shape()[3]);

    std::vector<std::unique_ptr<Mat>> grads;

    // 1. 计算 gW (卷积核梯度)
    // gy 形状: (N, OC, OH, OW) -> reshape 为 (N*OH*OW, OC)
    auto gy_reshaped = gy.reshape(Shape{N * OH * OW, OC});
    // 转置为 (OC, N*OH*OW)
    auto gy_T                 = gy_reshaped->transpose();
    const OriginMat &gy_T_mat = static_cast<const OriginMat &>(*gy_T);

    // 使用 im2col 将输入转换为列矩阵
    auto col = im2col_impl(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);
    // col 形状: (N*OH*OW, C*KH*KW)
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // gW = gy_T @ col，然后 reshape 为 (OC, C, KH, KW)
    auto gW_flat = gy_T_mat.matmul(col_mat);
    // gW_flat 形状: (OC, C*KH*KW)
    const OriginMat &gW_flat_mat = static_cast<const OriginMat &>(*gW_flat);
    auto gW                      = gW_flat_mat.reshape(Shape{OC, C, KH, KW});
    grads.push_back(std::move(gW));

    // 2. 计算 gb (偏置梯度)
    if (b != nullptr)
    {
        // gb = gy.sum(axis=(0, 2, 3))
        // gy 形状: (N, OC, OH, OW)
        // 方法：依次对维度2(OH), 2(OW), 0(N)求和
        auto gy_sum_h                 = gy.sum(2);  // sum over OH, shape: (N, OC, OW)
        const OriginMat &gy_sum_h_mat = static_cast<const OriginMat &>(*gy_sum_h);
        auto gy_sum_w                 = gy_sum_h_mat.sum(2);  // sum over OW, shape: (N, OC)
        const OriginMat &gy_sum_w_mat = static_cast<const OriginMat &>(*gy_sum_w);
        auto gb_mat                   = gy_sum_w_mat.sum(0);  // sum over N, shape: (OC,)
        auto gb                       = std::unique_ptr<OriginMat>(static_cast<OriginMat *>(gb_mat.release()));
        // 强制reshape到(OC,)，确保形状正确
        auto gb_reshaped = gb->reshape(Shape{OC});
        grads.push_back(std::move(gb_reshaped));
    }

    // 3. 计算 gx (输入梯度)
    // 将卷积核 reshape
    auto W_reshaped = W.reshape(Shape{OC, C * KH * KW});
    // gy_reshaped 形状: (N*OH*OW, OC)
    const OriginMat &gy_reshaped_mat = static_cast<const OriginMat &>(*gy_reshaped);
    const OriginMat &W_reshaped_mat  = static_cast<const OriginMat &>(*W_reshaped);
    // gx_col = gy_reshaped @ W_reshaped
    auto gx_col = gy_reshaped_mat.matmul(W_reshaped_mat);
    // gx_col 形状: (N*OH*OW, C*KH*KW)

    // 使用 col2im 转换回图像形状
    const OriginMat &gx_col_mat = static_cast<const OriginMat &>(*gx_col);
    auto gx = col2im_impl(gx_col_mat, x.shape(), std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride,
                          pad, true);
    grads.insert(grads.begin(), std::move(gx));  // 插入到开头，顺序为 {gx, gW, [gb]}

    return grads;
}

}  // namespace cpu
}  // namespace origin
