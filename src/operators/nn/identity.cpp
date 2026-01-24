#include "origin/operators/nn/identity.h"
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

/**
 * @brief Identity 算子的前向传播
 * @details Identity 算子实现恒等映射，直接返回输入，不做任何变换
 *
 * 核心作用：
 * 恒等映射：数学上等价于 f(x) = x，不改变数据。主要用于控制计算图结构，而非数据变换
 *
 * 行为说明：
 * 输入：直接返回该输入（使用 move 语义，避免拷贝）
 * 输入：返回最后一个输入（用于兼容 Detect 层等特殊场景）
 *
 * 应用场景：
 * 占位符算子：保持计算图结构，在需要时插入 Identity 而不改变数据流
 * 跳过某些层：在需要时插入 Identity 而不改变数据流
 * 处理 yolo_detect 层的多输入情况

 */
std::vector<Tensor> Identity::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.empty()))
    {
        THROW_RUNTIME_ERROR("Identity operator requires at least 1 input, but got 0");
    }

    // Identity 算子：直接返回输入
    // 对于 Detect 层，如果有多个输入，返回最后一个（通常是检测结果）
    // 但为了兼容性，如果有多个输入，返回所有输入
    if (xs.size() == 1)
    {
        // 单输入：直接返回该输入（使用 move 语义，避免拷贝）
        return std::vector<Tensor>{std::move(xs[0])};
    }
    else
    {
        // 多个输入时，返回最后一个（Detect 层的输出）
        // 用于兼容 Detect 层等特殊场景
        return std::vector<Tensor>{std::move(xs.back())};
    }
}

/**
 * @brief Identity 算子的反向传播
 * @details 反向传播时，梯度直接原样返回（也是恒等映射）
 *
 * 由于 Identity 算子的前向传播是恒等映射，其反向传播也是恒等映射：
 * 输入的梯度 = 输出的梯度（梯度直接传递，不做任何变换）
 */
std::vector<Tensor> Identity::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Identity backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &x  = this->inputs_[0];

    // 梯度直接原样返回（恒等映射）
    return std::vector<Tensor>{std::move(gy)};
}

Tensor identity(const Tensor &x)
{
    auto op = std::make_shared<Identity>();
    return (*op)(x);
}

}  // namespace functional
}  // namespace origin
