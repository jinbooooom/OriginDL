#include "origin/operators/shape/cat.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace functional
{

/**
 * @brief 检查两个 Shape 是否匹配（排除指定维度）
 * @param shape1 第一个形状
 * @param shape2 第二个形状
 * @param exclude_dim 要排除的维度索引
 * @return 如果除了 exclude_dim 外的所有维度都匹配，返回 true
 */
static inline bool shapes_match_except_dim(const Shape &shape1, const Shape &shape2, int exclude_dim)
{
    if (unlikely(shape1.size() != shape2.size()))
    {
        return false;
    }
    for (size_t d = 0; d < shape1.size(); ++d)
    {
        if (d != static_cast<size_t>(exclude_dim) && shape1[d] != shape2[d])
        {
            return false;
        }
    }
    return true;
}

/*
Cat::forward(const std::vector<Tensor> &xs): 包含所有输入（函数式语义）
Mat::cat(others, dim)：others 不包含 this（成员函数语义）
cuda::cat(inputs, dim)：inputs 包含所有输入（函数式语义）
这样每一步调用之间都需要做转换。
TODO：未来需要优化调用流程
*/
std::vector<Tensor> Cat::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.empty()))
    {
        THROW_RUNTIME_ERROR("Cat operator requires at least 1 input, but got 0");
    }

    if (xs.size() == 1)
    {
        return xs;  // 只有一个输入，直接返回
    }

    // 检查所有输入的形状（除了 dim_ 维度外应该相同）和设备
    const auto &first_shape = xs[0].shape();
    const auto &first_device = xs[0].device();

    for (size_t i = 1; i < xs.size(); ++i)
    {
        const auto &shape = xs[i].shape();
        if (unlikely(!shapes_match_except_dim(first_shape, shape, dim_)))
        {
            THROW_RUNTIME_ERROR("Cat forward: dimension mismatch at input {}, expected shape {} (excluding dim {}), "
                                "got shape {}",
                                i, first_shape.to_string(), dim_, shape.to_string());
        }

        if (unlikely(xs[i].device() != first_device))
        {
            THROW_RUNTIME_ERROR("Cat forward: device mismatch at input {}, expected {}, got {}", i,
                                first_device.to_string(), xs[i].device().to_string());
        }
    }

    std::vector<const Mat *> others(xs.size() - 1);
    for (size_t i = 1; i < xs.size(); ++i)
    {
        others[i - 1] = &mat(xs[i]);
    }
    std::unique_ptr<Mat> result_mat = mat(xs[0]).cat(others, dim_);

    Tensor result_tensor = convert_mat_to_tensor(std::move(result_mat));
    return {result_tensor};
}

std::vector<Tensor> Cat::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Cat backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy     = gys[0];
    auto &inputs = this->inputs_;

    std::vector<size_t> split_sizes(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        split_sizes[i] = inputs[i].shape()[dim_];
    }

    std::vector<std::unique_ptr<Mat>> gx_mats = mat(gy).split(split_sizes, dim_);
    std::vector<Tensor> gxs(gx_mats.size());
    for (size_t i = 0; i < gx_mats.size(); ++i)
    {
        gxs[i] = convert_mat_to_tensor(std::move(gx_mats[i]));
    }

    return gxs;
}

Tensor cat(const std::vector<Tensor> &xs, int dim)
{
    auto op      = std::make_shared<Cat>(dim);
    auto outputs = (*op)(xs);
    return outputs[0];
}

}  // namespace functional
}  // namespace origin
