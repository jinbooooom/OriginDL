#include "origin/operators/shape/cat.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
#ifdef WITH_CUDA
#    include "origin/mat/origin/cuda/cuda_ops.cuh"
#endif

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

    // 收集所有输入的 OriginMat 指针
    std::vector<const OriginMat *> input_mats;
    for (const auto &x : xs)
    {
        const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));
        input_mats.push_back(&x_mat);
    }

    // 根据设备类型调用对应的实现
    std::unique_ptr<Mat> result_mat;
    if (all_cuda)
    {
#ifdef WITH_CUDA
        result_mat = cuda::cat(input_mats, dim_);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        result_mat = cpu::cat(input_mats, dim_);
    }

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

    // 使用 mat 层的 split 函数分割梯度
    const OriginMat &gy_mat = static_cast<const OriginMat &>(mat(gy));
    std::vector<std::unique_ptr<Mat>> gx_mats;
    
    if (gy.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        gx_mats = cuda::split(gy_mat, output_shapes, dim_);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        gx_mats = cpu::split(gy_mat, output_shapes, dim_);
    }

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
