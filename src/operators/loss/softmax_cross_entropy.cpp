#include <algorithm>
#include <cmath>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/operators/activation/softmax.h"
#include "origin/operators/math/log.h"
#include "origin/operators/math/sum.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

std::vector<Tensor> SoftmaxCrossEntropy::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 2))
    {
        THROW_RUNTIME_ERROR("SoftmaxCrossEntropy operator requires exactly 2 inputs (x and target), but got {}",
                            xs.size());
    }

    auto &x      = xs[0];
    auto &target = xs[1];

    auto x_shape      = x.shape();
    auto target_shape = target.shape();

    // 验证输入形状
    if (unlikely(x_shape.size() != 2))
    {
        THROW_INVALID_ARG("SoftmaxCrossEntropy expects x to be 2D (N, C), but got shape {}", x_shape.to_string());
    }
    if (unlikely(target_shape.size() != 1))
    {
        THROW_INVALID_ARG("SoftmaxCrossEntropy expects target to be 1D (N,), but got shape {}",
                          target_shape.to_string());
    }
    if (unlikely(x_shape[0] != target_shape[0]))
    {
        THROW_INVALID_ARG("SoftmaxCrossEntropy: batch size mismatch. x has {} samples, target has {} samples",
                          x_shape[0], target_shape[0]);
    }

    size_t N = x_shape[0];  // batch size
    size_t C = x_shape[1];  // number of classes

    // 1. 计算 softmax: p = softmax(x)
    auto p = softmax(x, -1);  // 沿最后一个维度计算 softmax

    // 2. 使用 mat 层的 gather 提取 p[i][target[i]]
    const OriginMat &p_mat = static_cast<const OriginMat &>(mat(p));
    const OriginMat &target_mat = static_cast<const OriginMat &>(mat(target));
    
    // gather 从 p 中提取值：p.gather(target) 返回 (N,)
    auto p_selected_mat = p_mat.gather(target_mat);
    auto p_selected = convert_mat_to_tensor(std::move(p_selected_mat));

    // 3. 对提取的值取 log，并添加小的 epsilon 避免 log(0)
    // 先创建一个小的 epsilon 张量
    auto epsilon = Tensor({1e-8f}, Shape{}, dtype(DataType::kFloat32).device(x.device()));
    auto p_selected_safe = p_selected + epsilon;
    auto log_p = log(p_selected_safe);

    // 4. 计算 mean：sum 然后除以 N
    auto sum_log_p = sum(log_p, -1);  // sum 所有元素
    auto sum_value = sum_log_p.item<float>();
    float loss_value = -sum_value / static_cast<float>(N);

    // 创建标量损失张量
    auto loss = Tensor({loss_value}, Shape{}, dtype(DataType::kFloat32).device(x.device()));
    return std::vector<Tensor>{std::move(loss)};
}

std::vector<Tensor> SoftmaxCrossEntropy::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("SoftmaxCrossEntropy backward requires exactly 1 gradient, but got {}", gys.size());
    }

    // 梯度计算：gx = (softmax(x) - one_hot(target)) / N
    auto &x      = this->inputs_[0];
    auto &target = this->inputs_[1];
    auto &gy     = gys[0];

    auto x_shape = x.shape();
    size_t N     = x_shape[0];  // batch size
    size_t C     = x_shape[1];  // number of classes

    // 1. 计算 softmax(x)
    auto p = softmax(x, -1);

    // 2. 使用 mat 层的 one_hot 创建 one_hot(target) 编码
    const OriginMat &target_mat = static_cast<const OriginMat &>(mat(target));
    auto one_hot_mat = OriginMat::one_hot(target_mat, static_cast<int>(C));
    auto one_hot = convert_mat_to_tensor(std::move(one_hot_mat));

    // 3. 计算 gx = (softmax(x) - one_hot(target)) / N
    auto diff = p - one_hot;

    // 4. 应用梯度缩放（gy 通常是 1.0，但为了通用性，我们乘以它）
    auto gy_value = gy.item<float>();
    auto gx       = diff * (gy_value / static_cast<float>(N));

    // 5. target 不需要梯度（它是标签）
    auto gtarget = Tensor::zeros(target.shape(), dtype(DataType::kFloat32).device(target.device()));
    return std::vector<Tensor>{std::move(gx), std::move(gtarget)};
}

Tensor softmax_cross_entropy(const Tensor &x, const Tensor &target)
{
    auto op                    = std::make_shared<SoftmaxCrossEntropy>();
    std::vector<Tensor> inputs = {x, target};
    std::vector<Tensor> result = (*op)(inputs);
    return result[0];
}

}  // namespace functional
}  // namespace origin
