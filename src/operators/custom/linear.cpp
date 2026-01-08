#include "origin/operators/custom/linear.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/core/operator.h"

namespace origin
{

std::vector<Tensor> LinearOp::forward(const std::vector<Tensor> &xs)
{
    // xs[0] = x (输入), xs[1] = weight, xs[2] = bias (可选)
    size_t expected_inputs = use_bias_ ? 3 : 2;
    if (xs.size() != expected_inputs)
    {
        THROW_RUNTIME_ERROR("Linear operator requires {} inputs (x, weight{}), but got {}", 
                           expected_inputs, use_bias_ ? ", bias" : "", xs.size());
    }

    auto &x = xs[0];
    auto &weight = xs[1];
    const Tensor *bias = use_bias_ ? &xs[2] : nullptr;

    auto x_shape = x.shape();
    auto weight_shape = weight.shape();

    // 检查输入形状
    // x 应该是 (N, in_features) 或更高维度（会被展平）
    // weight 应该是 (out_features, in_features)
    
    // 如果 x 是 2D，直接使用
    // 如果 x 是更高维度，需要先展平（除了 batch 维度）
    Tensor x_flat = x;
    if (x_shape.size() > 2)
    {
        // 展平除 batch 维度外的所有维度
        size_t batch_size = x_shape[0];
        size_t features = 1;
        for (size_t i = 1; i < x_shape.size(); ++i)
        {
            features *= x_shape[i];
        }
        Shape flat_shape{batch_size, features};
        x_flat = reshape(x, flat_shape);
    }
    
    // 检查维度
    if (x_flat.shape().size() != 2)
    {
        THROW_RUNTIME_ERROR("Linear forward: x must be 2D after flattening, but got shape {}", 
                           x_flat.shape().to_string());
    }
    
    if (weight_shape.size() != 2)
    {
        THROW_RUNTIME_ERROR("Linear forward: weight must be 2D (out_features, in_features), but got shape {}", 
                           weight_shape.to_string());
    }
    
    size_t x_features = x_flat.shape()[1];
    size_t weight_out = weight_shape[0];
    size_t weight_in = weight_shape[1];
    
    if (x_features != weight_in)
    {
        THROW_RUNTIME_ERROR("Linear forward: feature mismatch - x has {} features, but weight expects {} features", 
                           x_features, weight_in);
    }
    
    if (weight_out != static_cast<size_t>(out_features_))
    {
        THROW_RUNTIME_ERROR("Linear forward: output feature mismatch - weight has {} out_features, but expected {}", 
                           weight_out, out_features_);
    }
    
    // 使用 matmul 进行矩阵乘法：x * weight^T
    // PyTorch 的 Linear 层：y = x * weight^T + bias
    // 所以我们需要先转置 weight
    auto weight_t = transpose(weight);
    
    // x: (N, in_features), weight_t: (in_features, out_features)
    // 结果: (N, out_features)
    auto y = mat_mul(x_flat, weight_t);
    
    // 添加偏置
    if (bias != nullptr)
    {
        // 广播偏置到 (N, out_features)
        Shape bias_shape{1, static_cast<size_t>(out_features_)};
        Shape y_shape = y.shape();
        auto bias_broadcast = broadcast_to(*bias, y_shape);
        y = y + bias_broadcast;
    }
    
    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> LinearOp::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Linear backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x = this->inputs_[0];
    auto &weight = this->inputs_[1];
    
    // 梯度计算
    // gx = gy * weight
    // gweight = gy^T * x
    // gbias = sum(gy, dim=0)
    
    // 展平 x（如果需要）
    Tensor x_flat = x;
    if (x.shape().size() > 2)
    {
        size_t batch_size = x.shape()[0];
        size_t features = 1;
        for (size_t i = 1; i < x.shape().size(); ++i)
        {
            features *= x.shape()[i];
        }
        Shape flat_shape{batch_size, features};
        x_flat = reshape(x, flat_shape);
    }
    
    // gx = gy * weight
    auto gx_flat = mat_mul(gy, weight);
    
    // 如果原始 x 是多维的，需要 reshape 回原始形状
    Tensor gx = gx_flat;
    if (x.shape().size() > 2)
    {
        gx = reshape(gx_flat, x.shape());
    }
    
    // gweight = gy^T * x_flat
    auto gy_t = transpose(gy);
    auto gweight = mat_mul(gy_t, x_flat);
    gweight = transpose(gweight);  // 转置回 (out_features, in_features)
    
    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(gweight);
    
    if (use_bias_)
    {
        // gbias = sum(gy, dim=0)
        auto gbias = sum(gy, 0);  // 沿 batch 维度求和
        outputs.push_back(gbias);
    }
    
    return outputs;
}

}  // namespace origin

