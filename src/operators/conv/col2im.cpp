#include "origin/operators/conv/col2im.h"
#include "origin/operators/conv/im2col.h"
#include "origin/utils/exception.h"
#include "origin/core/tensor.h"
#include <algorithm>
#include <vector>

namespace origin
{

std::vector<Tensor> Col2im::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 1)
    {
        THROW_RUNTIME_ERROR("Col2im operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &col = xs[0];
    auto col_shape = col.shape();

    // 检查输入形状
    if (input_shape_.size() != 4)
    {
        THROW_RUNTIME_ERROR("Col2im forward: input_shape must be 4D (N, C, H, W), but got shape {}",
                           input_shape_.to_string());
    }

    size_t N = input_shape_[0];
    size_t C = input_shape_[1];
    size_t H = input_shape_[2];
    size_t W = input_shape_[3];

    int KH = kernel_size_.first;
    int KW = kernel_size_.second;
    int SH = stride_.first;
    int SW = stride_.second;
    int PH = pad_.first;
    int PW = pad_.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    // 获取列矩阵数据
    auto col_data = col.to_vector<float>();

    // 如果是从矩阵形式转换，先 reshape
    std::vector<float> col_reshaped;
    if (to_matrix_)
    {
        // 从 (N*OH*OW, C*KH*KW) reshape 为 (N, C, KH, KW, OH, OW)
        if (col_shape.size() != 2 || col_shape[0] != static_cast<size_t>(N * OH * OW) ||
            col_shape[1] != static_cast<size_t>(C * KH * KW))
        {
            THROW_RUNTIME_ERROR("Col2im forward: invalid col shape {} for input_shape {}, kernel=({},{}), "
                               "stride=({},{}), pad=({},{})",
                               col_shape.to_string(), input_shape_.to_string(), KH, KW, SH, SW, PH, PW);
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
        if (col_shape.size() != 6 || col_shape[0] != N || col_shape[1] != C || col_shape[2] != static_cast<size_t>(KH) ||
            col_shape[3] != static_cast<size_t>(KW) || col_shape[4] != static_cast<size_t>(OH) ||
            col_shape[5] != static_cast<size_t>(OW))
        {
            THROW_RUNTIME_ERROR("Col2im forward: invalid col shape {} for input_shape {}, kernel=({},{}), "
                               "stride=({},{}), pad=({},{})",
                               col_shape.to_string(), input_shape_.to_string(), KH, KW, SH, SW, PH, PW);
        }
        col_reshaped = col_data;
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
                                size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W +
                                               h_idx * padded_W + w_idx;

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
                    size_t src_h = h + PH;
                    size_t src_w = w + PW;
                    size_t src_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + src_h * padded_W + src_w;
                    size_t dst_idx = n * C * H * W + c * H * W + h * W + w;
                    output_data[dst_idx] = img_data[src_idx];
                }
            }
        }
    }

    Shape out_shape{N, C, H, W};
    auto result = Tensor(output_data, out_shape, dtype(DataType::kFloat32).device(col.device()));

    std::vector<Tensor> outputs;
    outputs.push_back(result);
    return outputs;
}

std::vector<Tensor> Col2im::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Col2im backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // 使用 im2col 进行反向传播
    auto &gx = gys[0];
    auto gy = im2col(gx, kernel_size_, stride_, pad_, to_matrix_);

    std::vector<Tensor> outputs;
    outputs.push_back(gy);
    return outputs;
}

Tensor col2im(const Tensor &col, const Shape &input_shape, std::pair<int, int> kernel_size,
              std::pair<int, int> stride, std::pair<int, int> pad, bool to_matrix)
{
    auto op = std::make_shared<Col2im>(input_shape, kernel_size, stride, pad, to_matrix);
    return (*op)(col);
}

Tensor col2im(const Tensor &col, const Shape &input_shape, int kernel_size, int stride, int pad, bool to_matrix)
{
    return col2im(col, input_shape, pair(kernel_size), pair(stride), pair(pad), to_matrix);
}

}  // namespace origin

