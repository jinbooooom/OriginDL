#include "origin/operators/conv/im2col.h"
#include "origin/operators/conv/col2im.h"
#include "origin/utils/exception.h"
#include "origin/core/tensor.h"
#include <algorithm>
#include <vector>

namespace origin
{

std::vector<Tensor> Im2col::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Im2col operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];
    auto x_shape = x.shape();

    // 检查输入形状：必须是 (N, C, H, W)
    if (x_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Im2col forward: input must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    input_shape_ = x_shape;

    size_t N = x_shape[0];
    size_t C = x_shape[1];
    size_t H = x_shape[2];
    size_t W = x_shape[3];

    int KH = kernel_size_.first;
    int KW = kernel_size_.second;
    int SH = stride_.first;
    int SW = stride_.second;
    int PH = pad_.first;
    int PW = pad_.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    if (OH <= 0 || OW <= 0)
    {
        THROW_RUNTIME_ERROR("Im2col forward: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
                           "stride=({},{}), pad=({},{})",
                           OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 获取输入数据
    auto x_data = x.to_vector<float>();

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
                    size_t dst_h = h + PH;
                    size_t dst_w = w + PW;
                    size_t dst_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + dst_h * padded_W + dst_w;
                    padded_img[dst_idx] = x_data[src_idx];
                }
            }
        }
    }

    // 创建列矩阵
    if (to_matrix_)
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
        auto result = Tensor(col_data, out_shape, dtype(DataType::kFloat32).device(x.device()));

        std::vector<Tensor> outputs;
        outputs.push_back(result);
        return outputs;
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

        Shape out_shape{N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), static_cast<size_t>(OH),
                       static_cast<size_t>(OW)};
        auto result = Tensor(col_data, out_shape, dtype(DataType::kFloat32).device(x.device()));

        std::vector<Tensor> outputs;
        outputs.push_back(result);
        return outputs;
    }
}

std::vector<Tensor> Im2col::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Im2col backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // 使用 col2im 进行反向传播
    auto &gy = gys[0];
    auto gx = col2im(gy, input_shape_, kernel_size_, stride_, pad_, to_matrix_);

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    return outputs;
}

Tensor im2col(const Tensor &x, std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> pad,
              bool to_matrix)
{
    auto op = std::make_shared<Im2col>(kernel_size, stride, pad, to_matrix);
    return (*op)(x);
}

Tensor im2col(const Tensor &x, int kernel_size, int stride, int pad, bool to_matrix)
{
    return im2col(x, pair(kernel_size), pair(stride), pair(pad), to_matrix);
}

}  // namespace origin

