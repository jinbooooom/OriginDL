#include "origin/operators/nn/embedding.h"
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
namespace origin
{
namespace functional
{
std::vector<Tensor> EmbeddingOp::forward(const std::vector<Tensor> &xs)
{
    size_t expected_inputs = 2;
    if (unlikely(xs.size() != expected_inputs))
    {
        THROW_RUNTIME_ERROR("Embedding operator requires {} inputs (x, weight), but got {}", expected_inputs,
                            xs.size());
    }
    auto &x           = xs[0];
    auto &weight      = xs[1];
    auto x_shape      = x.shape();
    auto weight_shape = weight.shape();

    // 如果 vocab_size_ 或 embedding_dim_ 为 0，从 weight 中获取
    if (vocab_size_ == 0 || embedding_dim_ == 0)
    {
        vocab_size_ = static_cast<int>(weight_shape[0]);
        embedding_dim_ = static_cast<int>(weight_shape[1]);
    }

    // Embedding 层应该保持输入的所有维度，在最后添加 embedding_dim 维度
    // 不需要对输入进行 reshape，CPU 实现可以处理任意维度的输入

    // 检查权重维度
    if (unlikely(weight_shape.size() != 2))
    {
        THROW_RUNTIME_ERROR("Embedding forward: weight must be 2D (vocab_size, embedding_dim), but got shape {}",
                            weight_shape.to_string());
    }

    std::unique_ptr<Mat> result;
    const Mat &x_mat     = mat(x);
    const Mat &vocab_mat = mat(weight);
    result               = x_mat.embedding(vocab_mat);

    if (x.requires_grad())
    {
        auto indices_mat_unique = x_mat.clone();
        indices_                = convert_mat_to_tensor(std::move(indices_mat_unique));
    }

    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> EmbeddingOp::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Embedding backward requires exactly 1 gradient, but got {}", gys.size());
    }
    auto &gy = gys[0];
    if (unlikely(indices_.elements() == 0))
    {
        THROW_RUNTIME_ERROR(
            "Embedding backward: indices_ is not initialized. This should not happen when requires_grad=true");
    }
    const Mat &gy_mat           = mat(gy);
    const Mat &indices_mat      = mat(indices_);
    std::unique_ptr<Mat> result = gy_mat.embedding_back(indices_mat, vocab_size_, embedding_dim_);

    // 第一个梯度：对索引 x 的梯度（索引是整数，不需要梯度，返回零张量）
    Tensor gx_indices = Tensor::zeros(indices_.shape(), dtype(indices_.dtype()).device(indices_.device()));

    // 第二个梯度：对权重 weight 的梯度
    auto gweight = convert_mat_to_tensor(std::move(result));

    return std::vector<Tensor>{std::move(gx_indices), std::move(gweight)};
}

Tensor embedding(const Tensor &x, const Tensor &weight, int vocab_size_, int embedding_dim_)
{
    auto op = std::make_shared<EmbeddingOp>(
        vocab_size_, embedding_dim_);  // vocab_size 和 embedding_dim 会在 forward 中从 weight 获取
    return (*op)({x, weight})[0];
}

}  // namespace functional
}  // namespace origin