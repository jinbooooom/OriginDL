#include "origin/nn/layers/embedding.h"
#include <cmath>
#include <vector>
#include "origin/core/operator.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
namespace origin
{
namespace nn
{
Embedding::Embedding(int vocab_size, int embedding_dim) : Embedding(vocab_size, embedding_dim, DataType::kFloat32) {}

Embedding::Embedding(int vocab_size, int embedding_dim, DataType dtype)
    : vocab_size_(vocab_size), embedding_dim_(embedding_dim), dtype_(dtype)
{
    // 验证 dtype 不能是整数类型
    if (unlikely(dtype == DataType::kInt8 || dtype == DataType::kInt32 ||
                 dtype == DataType::kInt64 || dtype == DataType::kUInt8))
    {
        THROW_INVALID_ARG("Embedding layer dtype must be floating-point type (float32 or float64), got {}",
                         dtype_to_string(dtype));
    }

    weight_ = init_weight();
    register_parameter("weight", weight_);
}

Parameter Embedding::init_weight()
{
    // Xavier 初始化
    float scale        = std::sqrt(1.0f / static_cast<float>(embedding_dim_));
    auto weight_tensor = Tensor::randn(Shape{static_cast<size_t>(vocab_size_), static_cast<size_t>(embedding_dim_)},
                                       TensorOptions(dtype_).requires_grad(true));
    auto scaled_weight = weight_tensor * Scalar(scale);
    Parameter w(scaled_weight);
    return w;
}

void Embedding::init_parameters()
{
    weight_ = init_weight();
}

void Embedding::reset_parameters()
{
    weight_ = init_weight();
}

Tensor Embedding::forward(const Tensor &input)
{
    // 输入验证
    if (unlikely(input.dtype() != DataType::kInt32))
    {
        THROW_INVALID_ARG("Embedding input must be int32, got {}", dtype_to_string(input.dtype()));
    }

    auto w_shape = weight_.shape();
    if (unlikely(w_shape.elements() == 0))
    {
        THROW_RUNTIME_ERROR("Embedding: Weight is empty in forward! weight_.shape() = {}", w_shape.to_string());
    }
    auto op = std::make_shared<functional::EmbeddingOp>(vocab_size_, embedding_dim_);
    std::vector<Tensor> inputs;
    inputs.push_back(input);
    inputs.push_back(static_cast<Tensor &>(weight_));
    auto outputs = (*op)(inputs);

    return outputs[0];
}
}  // namespace nn
}  // namespace origin