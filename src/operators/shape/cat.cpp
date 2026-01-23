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

    // 检查所有输入的形状（除了 dim_ 维度外应该相同）
    auto first_shape = xs[0].shape();
    Device device    = xs[0].device();
    bool all_cuda    = (device.type() == DeviceType::kCUDA);

    for (size_t i = 1; i < xs.size(); ++i)
    {
        auto shape = xs[i].shape();
        if (unlikely(shape.size() != first_shape.size()))
        {
            THROW_RUNTIME_ERROR("Cat forward: all inputs must have same number of dimensions");
        }

        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (unlikely(d != static_cast<size_t>(dim_) && shape[d] != first_shape[d]))
            {
                THROW_RUNTIME_ERROR("Cat forward: dimension {} mismatch: {} vs {}", d, shape[d], first_shape[d]);
            }
        }

        // 检查是否所有输入都在 CUDA 上
        if (xs[i].device().type() != DeviceType::kCUDA)
        {
            all_cuda = false;
        }
    }

    // 计算输出形状
    Shape output_shape = first_shape;
    int total_dim_size = 0;
    for (const auto &x : xs)
    {
        total_dim_size += x.shape()[dim_];
    }
    output_shape[dim_] = total_dim_size;

    // 收集所有输入的 Mat 指针
    std::vector<const Mat *> input_mats;
    for (const auto &x : xs)
    {
        input_mats.push_back(&mat(x));
    }

    // 使用 Mat 接口的静态方法
    std::unique_ptr<Mat> result_mat = Mat::cat(input_mats, dim_);

    // 转换为 Tensor
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

    // 收集所有输入的形状
    std::vector<Shape> output_shapes;
    output_shapes.reserve(inputs.size());
    for (const auto &x : inputs)
    {
        output_shapes.push_back(x.shape());
    }

    // 使用 Mat 接口的静态方法分割梯度
    std::vector<std::unique_ptr<Mat>> gx_mats = Mat::split(mat(gy), output_shapes, dim_);

    // 转换为 Tensor
    std::vector<Tensor> gxs;
    gxs.reserve(gx_mats.size());
    for (auto &gx_mat : gx_mats)
    {
        gxs.push_back(convert_mat_to_tensor(std::move(gx_mat)));
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
