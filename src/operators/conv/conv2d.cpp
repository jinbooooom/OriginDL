#include "origin/operators/conv/conv2d.h"
#include "origin/utils/exception.h"
#include "origin/core/tensor.h"
#include "origin/core/operator.h"
#include <vector>

namespace origin
{

std::vector<Tensor> Conv2d::forward(const std::vector<Tensor> &xs)
{
    // xs[0] = x (输入), xs[1] = W (卷积核), xs[2] = b (偏置，可选)
    if (xs.size() < 2 || xs.size() > 3)
    {
        THROW_RUNTIME_ERROR("Conv2d operator requires 2 or 3 inputs (x, W, [b]), but got {}", xs.size());
    }

    auto &x = xs[0];
    auto &W = xs[1];
    const Tensor *b = (xs.size() == 3) ? &xs[2] : nullptr;

    auto x_shape = x.shape();
    auto W_shape = W.shape();

    // 检查输入形状
    if (x_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: x must be 4D (N, C, H, W), but got shape {}", x_shape.to_string());
    }

    if (W_shape.size() != 4)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: W must be 4D (OC, C, KH, KW), but got shape {}", W_shape.to_string());
    }

    size_t N = x_shape[0];
    size_t C = x_shape[1];
    size_t H = x_shape[2];
    size_t W_in = x_shape[3];

    size_t OC = W_shape[0];
    size_t C_in = W_shape[1];
    size_t KH = W_shape[2];
    size_t KW = W_shape[3];

    // 检查通道数是否匹配
    if (C != C_in)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: channel mismatch - x has {} channels, but W expects {} channels", C,
                           C_in);
    }

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), static_cast<int>(KH), stride_.first, pad_.first);
    int OW = get_conv_outsize(static_cast<int>(W_in), static_cast<int>(KW), stride_.second, pad_.second);

    if (OH <= 0 || OW <= 0)
    {
        THROW_RUNTIME_ERROR("Conv2d forward: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
                           "stride=({},{}), pad=({},{})",
                           OH, OW, H, W_in, KH, KW, stride_.first, stride_.second, pad_.first, pad_.second);
    }

    // 1. 使用 im2col 将输入转换为列矩阵
    auto col = im2col(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride_, pad_, true);
    // col 形状: (N*OH*OW, C*KH*KW)

    // 2. 将卷积核 reshape 为 (OC, C*KH*KW) 并转置为 (C*KH*KW, OC)
    auto W_reshaped = W.reshape(Shape{OC, C * KH * KW});
    auto W_T = W_reshaped.transpose();
    // W_T 形状: (C*KH*KW, OC)

    // 3. 执行矩阵乘法: col @ W_T
    auto y_flat = mat_mul(col, W_T);
    // y_flat 形状: (N*OH*OW, OC)

    // 4. 添加偏置（如果存在）
    Tensor y_with_bias;
    if (b != nullptr)
    {
        auto b_shape = b->shape();
        if (b_shape.size() != 1 || b_shape[0] != OC)
        {
            THROW_RUNTIME_ERROR("Conv2d forward: b must be 1D with size {}, but got shape {}", OC, b_shape.to_string());
        }

        // 广播偏置: (OC,) -> (N*OH*OW, OC)
        auto b_broadcast = broadcast_to(*b, Shape{N * OH * OW, OC});
        y_with_bias = y_flat + b_broadcast;
    }
    else
    {
        y_with_bias = y_flat;
    }

    // 5. Reshape 为最终输出形状 (N, OC, OH, OW)
    auto y = y_with_bias.reshape(Shape{N, static_cast<size_t>(OH), static_cast<size_t>(OW), OC});
    // 需要 transpose 为 (N, OC, OH, OW)
    // 由于没有直接的 transpose(axes)，我们需要手动处理
    auto y_data = y.to_vector<float>();
    std::vector<float> y_transposed(N * OC * OH * OW);
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t oc = 0; oc < OC; ++oc)
        {
            for (int oh = 0; oh < OH; ++oh)
            {
                for (int ow = 0; ow < OW; ++ow)
                {
                    size_t src_idx = n * OH * OW * OC + oh * OW * OC + ow * OC + oc;
                    size_t dst_idx = n * OC * OH * OW + oc * OH * OW + oh * OW + ow;
                    y_transposed[dst_idx] = y_data[src_idx];
                }
            }
        }
    }

    Shape out_shape{N, OC, static_cast<size_t>(OH), static_cast<size_t>(OW)};
    auto result = Tensor(y_transposed, out_shape, dtype(DataType::kFloat32).device(x.device()));

    std::vector<Tensor> outputs;
    outputs.push_back(result);
    return outputs;
}

std::vector<Tensor> Conv2d::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Conv2d backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x = this->inputs_[0];
    auto &W = this->inputs_[1];
    const Tensor *b = (this->inputs_.size() == 3) ? &this->inputs_[2] : nullptr;

    auto x_shape = x.shape();
    auto W_shape = W.shape();
    auto gy_shape = gy.shape();

    size_t N = x_shape[0];
    size_t C = x_shape[1];
    size_t H = x_shape[2];
    size_t W_in = x_shape[3];

    size_t OC = W_shape[0];
    size_t KH = W_shape[2];
    size_t KW = W_shape[3];

    int OH = static_cast<int>(gy_shape[2]);
    int OW = static_cast<int>(gy_shape[3]);

    // 1. 计算 gb (偏置梯度)
    Tensor gb;
    if (b != nullptr)
    {
        // gb = gy.sum(axis=(0, 2, 3))
        // 先对 OH 维度求和 (axis=2)
        auto gy_sum_h = sum(gy, 2);  // sum over OH, shape: (N, OC, OW)
        // 再对 OW 维度求和 (axis=2，因为 OH 已经被移除)
        auto gy_sum_w = sum(gy_sum_h, 2);  // sum over OW, shape: (N, OC)
        // 最后对 N 维度求和 (axis=0)
        auto gy_sum_n = sum(gy_sum_w, 0);  // sum over N, shape: (OC,)
        gb = gy_sum_n;
    }

    // 2. 计算 gW (卷积核梯度)
    // gy 形状: (N, OC, OH, OW) -> reshape 为 (N*OH*OW, OC)
    auto gy_reshaped = gy.reshape(Shape{N * OH * OW, OC});
    // 转置为 (OC, N*OH*OW)
    auto gy_T = gy_reshaped.transpose();

    // 使用 im2col 将输入转换为列矩阵
    auto col = im2col(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride_, pad_, true);
    // col 形状: (N*OH*OW, C*KH*KW)

    // gW = gy_T @ col，然后 reshape 为 (OC, C, KH, KW)
    auto gW_flat = mat_mul(gy_T, col);
    // gW_flat 形状: (OC, C*KH*KW)
    auto gW = gW_flat.reshape(Shape{OC, C, KH, KW});

    // 3. 计算 gx (输入梯度)
    // 将卷积核 reshape 并转置
    auto W_reshaped = W.reshape(Shape{OC, C * KH * KW});
    // gy_reshaped 形状: (N*OH*OW, OC)
    // gx_col = gy_reshaped @ W_reshaped
    auto gx_col = mat_mul(gy_reshaped, W_reshaped);
    // gx_col 形状: (N*OH*OW, C*KH*KW)

    // 使用 col2im 转换回图像形状
    auto gx = col2im(gx_col, x_shape, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride_, pad_, true);

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(gW);
    if (b != nullptr)
    {
        outputs.push_back(gb);
    }
    return outputs;
}

Tensor conv2d(const Tensor &x, const Tensor &W, const Tensor *b, std::pair<int, int> stride, std::pair<int, int> pad)
{
    auto op = std::make_shared<Conv2d>(stride, pad);
    if (b != nullptr)
    {
        return (*op)({x, W, *b})[0];
    }
    else
    {
        return (*op)({x, W})[0];
    }
}

Tensor conv2d(const Tensor &x, const Tensor &W, const Tensor *b, int stride, int pad)
{
    return conv2d(x, W, b, pair(stride), pair(pad));
}

}  // namespace origin

