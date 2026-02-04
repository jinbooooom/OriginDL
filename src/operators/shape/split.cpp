#include "origin/operators/shape/split.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"
#include <algorithm>

namespace origin
{
namespace functional
{

std::vector<Tensor> Split::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Split operator requires exactly 1 input, but got {}", xs.size());
    }

    const auto &x = xs[0];
    const auto &input_shape = x.shape();

    if (unlikely(dim_ < 0 || static_cast<size_t>(dim_) >= input_shape.size()))
    {
        THROW_INVALID_ARG("Split: dim {} is out of range for input shape {}", dim_, input_shape.to_string());
    }

    const size_t dim_size = input_shape[dim_];

    // 计算分割大小列表
    std::vector<size_t> actual_split_sizes;
    if (split_sizes_.size() == 1 && split_sizes_[0] > 0)
    {
        // 按固定大小分割模式
        // 维度 dim 上有 dim_size 个元素，按 split_sizes_[0] 大小分割
        // 比如 dim_size = 10，split_size = 3，则分割的张量数 num_splits 为 4，actual_split_sizes = [3, 3, 3, 1]
        size_t split_size = split_sizes_[0];
        size_t num_splits = (dim_size + split_size - 1) / split_size;  // 向上取整
        actual_split_sizes.resize(num_splits);
        for (size_t i = 0; i < num_splits; ++i)
        {
            actual_split_sizes[i] = (i < num_splits - 1) ? split_size : dim_size - (num_splits - 1) * split_size;
        }
    }
    else
    {
        // 按大小列表分割模式
        actual_split_sizes = split_sizes_;
        size_t total_size = 0;
        for (size_t size : actual_split_sizes)
        {
            total_size += size;
        }
        
        // 验证分割大小总和是否等于维度大小
        if (unlikely(total_size != dim_size))
        {
            THROW_INVALID_ARG("Split: sum of split_sizes {} does not match dimension size {} at dim {}",
                             total_size, dim_size, dim_);
        }
    }

    std::vector<std::unique_ptr<Mat>> result_mats = mat(x).split(actual_split_sizes, dim_);
    std::vector<Tensor> results(result_mats.size());
    for (size_t i = 0; i < result_mats.size(); ++i)
    {
        results[i] = convert_mat_to_tensor(std::move(result_mats[i]));
    }

    return results;
}

std::vector<Tensor> Split::backward(const std::vector<Tensor> &gys)
{
    // Split 的反向是 cat
    // 将所有分割后的梯度拼接回原始形状
    
    if (unlikely(gys.empty()))
    {
        THROW_RUNTIME_ERROR("Split backward requires at least 1 gradient, but got 0");
    }

    if (gys.size() == 1)
    {
        return gys;
    }

    // 检查所有梯度的形状（除了 dim_ 维度外应该相同）和设备
    const auto &first_shape = gys[0].shape();
    const auto &first_device = gys[0].device();

    for (size_t i = 1; i < gys.size(); ++i)
    {
        const auto &shape = gys[i].shape();
        if (unlikely(shape.size() != first_shape.size()))
        {
            THROW_RUNTIME_ERROR("Split backward: all gradients must have same number of dimensions");
        }

        // 除了 dim_ 维度外，所有梯度的形状应该相同，否则无法拼接
        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (d != static_cast<size_t>(dim_) && shape[d] != first_shape[d])
            {
                THROW_RUNTIME_ERROR("Split backward: dimension {} mismatch: {} vs {}", d, shape[d], first_shape[d]);
            }
        }

        if (unlikely(gys[i].device() != first_device))
        {
            THROW_RUNTIME_ERROR("Split backward: device mismatch at gradient {}, expected {}, got {}", i,
                                first_device.to_string(), gys[i].device().to_string());
        }
    }

    std::vector<const Mat *> others(gys.size() - 1);
    for (size_t i = 1; i < gys.size(); ++i)
    {
        others[i - 1] = &mat(gys[i]);
    }
    std::unique_ptr<Mat> result_mat = mat(gys[0]).cat(others, dim_);
    Tensor result_tensor = convert_mat_to_tensor(std::move(result_mat));
    return {result_tensor};
}

std::vector<Tensor> split(const Tensor &x, SizeArrayRef split_sizes, int dim)
{
    // SizeArrayRef 直接传递给 Split 构造函数
    // Split 构造函数内部会调用 to_vector() 拷贝数据到 split_sizes_ 成员变量
    auto op = std::make_shared<Split>(split_sizes, dim);
    std::vector<Tensor> inputs = {x};
    return (*op)(inputs);
}

std::vector<Tensor> split(const Tensor &x, size_t split_size, int dim)
{
    auto op = std::make_shared<Split>(split_size, dim);
    std::vector<Tensor> inputs = {x};
    return (*op)(inputs);
}

}  // namespace functional
}  // namespace origin
