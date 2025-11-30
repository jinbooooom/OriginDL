#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/type_promotion.h"
#include "origin/utils/exception.h"

namespace origin
{

std::vector<Tensor> MatMul::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != 2)
    {
        THROW_RUNTIME_ERROR("MatMul operator requires exactly 2 inputs, but got {}", xs.size());
    }

    // 检查类型是否匹配，如果不匹配则进行类型提升
    Tensor x0, x1;
    if (xs[0].dtype() != xs[1].dtype())
    {
        // 自动类型提升
        DataType promoted_type = promote_types_rule(xs[0].dtype(), xs[1].dtype());
        x0                     = xs[0].dtype() == promoted_type ? xs[0] : xs[0].to(promoted_type);
        x1                     = xs[1].dtype() == promoted_type ? xs[1] : xs[1].to(promoted_type);
    }
    else
    {
        x0 = xs[0];
        x1 = xs[1];
    }

    // 处理维度：确保两个输入至少是2维的
    auto shape0 = x0.shape();
    auto shape1 = x1.shape();

    // 检查shape是否有效
    if (shape0.elements() == 0 || shape1.elements() == 0)
    {
        THROW_RUNTIME_ERROR("MatMul forward: input shapes invalid - xs[0].shape() = {}, xs[1].shape() = {}",
                            shape0.to_string(), shape1.to_string());
    }

    // 自动展开：将0维或1维张量展开为2维
    // 0维 -> {1, 1}
    // 1维 -> {1, n} 或 {n, 1} (根据上下文)
    if (shape0.size() == 0)
    {
        // 0维标量：展开为 {1, 1}
        x0     = x0.reshape(Shape{1, 1});
        shape0 = x0.shape();
    }
    else if (shape0.size() == 1)
    {
        // 1维向量：展开为 {1, n}（行向量）
        x0     = x0.reshape(Shape{1, shape0[0]});
        shape0 = x0.shape();
    }

    if (shape1.size() == 0)
    {
        // 0维标量：展开为 {1, 1}
        x1     = x1.reshape(Shape{1, 1});
        shape1 = x1.shape();
    }
    else if (shape1.size() == 1)
    {
        // 1维向量：展开为 {n, 1}（列向量）
        // 注意：如果x0是{batch, 1}，x1应该是{1, out_features}
        // 但如果x1是1维，我们假设它是列向量，展开为{n, 1}
        x1     = x1.reshape(Shape{shape1[0], 1});
        shape1 = x1.shape();
    }

    // 确保两个张量至少是2维的
    if (shape0.size() < 2 || shape1.size() < 2)
    {
        THROW_RUNTIME_ERROR(
            "MatMul forward: after reshape, shapes must be at least 2D - x0.shape() = {}, x1.shape() = {}",
            shape0.to_string(), shape1.to_string());
    }

    // 检查矩阵乘法的维度兼容性
    if (shape0.size() == 2 && shape1.size() == 2)
    {
        if (shape0[1] != shape1[0])
        {
            THROW_RUNTIME_ERROR(
                "MatMul forward: dimension mismatch - x0.shape() = {}, x1.shape() = {}, x0[1]={} != x1[0]={}",
                shape0.to_string(), shape1.to_string(), shape0[1], shape1[0]);
        }
    }
    else if (shape0.size() == 3 && shape1.size() == 2)
    {
        // 批量矩阵乘法：{batch, m, k} x {k, n} -> {batch, m, n}
        if (shape0[2] != shape1[0])
        {
            THROW_RUNTIME_ERROR(
                "MatMul forward: dimension mismatch - x0.shape() = {}, x1.shape() = {}, x0[2]={} != x1[0]={}",
                shape0.to_string(), shape1.to_string(), shape0[2], shape1[0]);
        }
    }
    else
    {
        THROW_RUNTIME_ERROR("MatMul forward: unsupported shape combination - x0.shape() = {}, x1.shape() = {}",
                            shape0.to_string(), shape1.to_string());
    }

    // 执行矩阵乘法
    auto result = mat(x0).matmul(mat(x1));
    auto y      = convert_mat_to_tensor(std::move(result));

    std::vector<Tensor> outputs;
    outputs.push_back(y);
    return outputs;
}

std::vector<Tensor> MatMul::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("MatMul backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // TODO: 未来需要在backward中也实现类型提升逻辑

    // 获取输入张量并处理维度（与forward中的处理保持一致）
    Tensor x_tensor  = this->inputs_[0];
    Tensor w_tensor  = this->inputs_[1];
    Tensor gy_tensor = gys[0];

    // 处理维度：确保至少是2维（与forward中的逻辑一致）
    auto x_shape = x_tensor.shape();
    auto w_shape = w_tensor.shape();

    // 自动展开：将0维或1维张量展开为2维
    if (x_shape.size() == 0)
    {
        x_tensor = x_tensor.reshape(Shape{1, 1});
        x_shape  = x_tensor.shape();
    }
    else if (x_shape.size() == 1)
    {
        x_tensor = x_tensor.reshape(Shape{1, x_shape[0]});
        x_shape  = x_tensor.shape();
    }

    if (w_shape.size() == 0)
    {
        w_tensor = w_tensor.reshape(Shape{1, 1});
        w_shape  = w_tensor.shape();
    }
    else if (w_shape.size() == 1)
    {
        w_tensor = w_tensor.reshape(Shape{w_shape[0], 1});
        w_shape  = w_tensor.shape();
    }

    // 确保至少是2维
    if (x_shape.size() < 2 || w_shape.size() < 2)
    {
        THROW_RUNTIME_ERROR(
            "MatMul backward: after reshape, shapes must be at least 2D - x.shape() = {}, w.shape() = {}",
            x_shape.to_string(), w_shape.to_string());
    }

    // 获取Mat引用
    auto &x  = mat(x_tensor);
    auto &w  = mat(w_tensor);
    auto &gy = mat(gy_tensor);

    // 使用抽象层进行梯度计算
    auto w_T = w.transpose();
    auto x_T = x.transpose();

    auto gx_result = gy.matmul(*w_T);
    auto gw_result = x_T->matmul(gy);

    auto gx = convert_mat_to_tensor(std::move(gx_result));
    auto gw = convert_mat_to_tensor(std::move(gw_result));

    std::vector<Tensor> outputs;
    outputs.push_back(gx);
    outputs.push_back(gw);
    return outputs;
}

Tensor matmul(const std::vector<Tensor> &xs)
{
    auto op = std::make_shared<MatMul>();
    return (*op)(xs)[0];
}

Tensor matmul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul({lhs, rhs});
}

Tensor mat_mul(const Tensor &lhs, const Tensor &rhs)
{
    return matmul(lhs, rhs);
}

}  // namespace origin