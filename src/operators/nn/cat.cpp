#include "origin/operators/nn/cat.h"
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"
#include "origin/mat/origin/origin_mat.h"

namespace origin
{

std::vector<Tensor> Cat::forward(const std::vector<Tensor> &xs)
{
    if (xs.empty())
    {
        THROW_RUNTIME_ERROR("Cat operator requires at least 1 input, but got 0");
    }

    if (xs.size() == 1)
    {
        return xs;  // 只有一个输入，直接返回
    }

    // 检查所有输入的形状（除了 dim_ 维度外应该相同）
    auto first_shape = xs[0].shape();
    for (size_t i = 1; i < xs.size(); ++i)
    {
        auto shape = xs[i].shape();
        if (shape.size() != first_shape.size())
        {
            THROW_RUNTIME_ERROR("Cat forward: all inputs must have same number of dimensions");
        }
        
        for (size_t d = 0; d < shape.size(); ++d)
        {
            if (d != static_cast<size_t>(dim_) && shape[d] != first_shape[d])
            {
                THROW_RUNTIME_ERROR("Cat forward: dimension {} mismatch: {} vs {}", 
                                   d, shape[d], first_shape[d]);
            }
        }
    }

    // 实现 cat 操作：在指定维度上拼接
    // 计算输出形状
    Shape output_shape = first_shape;
    int total_dim_size = 0;
    for (const auto &x : xs)
    {
        total_dim_size += x.shape()[dim_];
    }
    output_shape[dim_] = total_dim_size;
    
    // 优化：先一次性将所有输入复制到 CPU，避免在循环中多次复制
    // 确定设备类型（使用第一个输入的设备）
    Device device = xs[0].device();
    
    // 预先将所有输入数据复制到 CPU（无论原始设备是什么，都在 CPU 上进行拼接）
    std::vector<std::vector<float>> input_data_vecs;
    for (const auto &x : xs)
    {
        input_data_vecs.push_back(x.to_vector<float>());
    }
    
    // 收集所有数据并拼接
    std::vector<float> output_data(output_shape.elements());
    
    // 计算每个维度的步长
    auto compute_strides = [](const Shape &shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    };
    
    auto output_strides = compute_strides(output_shape);
    
    // 对每个输出位置，确定应该从哪个输入读取
    for (size_t i = 0; i < output_data.size(); ++i)
    {
        // 计算输出位置的坐标
        std::vector<int> coords(output_shape.size());
        size_t idx = i;
        for (size_t d = 0; d < output_shape.size(); ++d)
        {
            coords[d] = idx / output_strides[d];
            idx %= output_strides[d];
        }
        
        // 确定应该从哪个输入读取
        int input_idx = 0;
        int offset_in_dim = 0;
        int current_offset = coords[dim_];
        
        for (const auto &x : xs)
        {
            int dim_size = x.shape()[dim_];
            if (current_offset < offset_in_dim + dim_size)
            {
                break;
            }
            offset_in_dim += dim_size;
            input_idx++;
        }
        
        // 计算在输入中的坐标
        std::vector<int> input_coords = coords;
        input_coords[dim_] = current_offset - offset_in_dim;
        
        // 从输入中读取数据
        const auto &input = xs[input_idx];
        auto input_shape = input.shape();
        auto input_strides = compute_strides(input_shape);
        
        size_t input_pos = 0;
        for (size_t d = 0; d < input_coords.size(); ++d)
        {
            input_pos += input_coords[d] * input_strides[d];
        }
        
        // 使用预先复制的数据
        output_data[i] = input_data_vecs[input_idx][input_pos];
    }
    
    // 创建输出 tensor，使用原始设备
    auto y = Tensor(output_data, output_shape, dtype(xs[0].dtype()).device(device));
    std::vector<Tensor> result;
    result.push_back(y);
    return result;
}

std::vector<Tensor> Cat::backward(const std::vector<Tensor> &gys)
{
    if (gys.size() != 1)
    {
        THROW_RUNTIME_ERROR("Cat backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];
    auto &inputs = this->inputs_;
    
    // 将梯度分割回各个输入
    std::vector<Tensor> gxs;
    auto gy_data = gy.to_vector<float>();
    auto gy_shape = gy.shape();
    
    // 计算步长
    auto compute_strides = [](const Shape &shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    };
    
    auto gy_strides = compute_strides(gy_shape);
    int offset_in_dim = 0;
    
    for (const auto &x : inputs)
    {
        auto x_shape = x.shape();
        auto x_strides = compute_strides(x_shape);
        std::vector<float> gx_data(x_shape.elements());
        
        // 从 gy 中提取对应的部分
        for (size_t i = 0; i < gx_data.size(); ++i)
        {
            // 计算在输入中的坐标
            std::vector<int> coords(x_shape.size());
            size_t idx = i;
            for (size_t d = 0; d < x_shape.size(); ++d)
            {
                coords[d] = idx / x_strides[d];
                idx %= x_strides[d];
            }
            
            // 计算在输出中的坐标
            std::vector<int> gy_coords = coords;
            gy_coords[dim_] += offset_in_dim;
            
            // 计算在 gy 中的位置
            size_t gy_pos = 0;
            for (size_t d = 0; d < gy_coords.size(); ++d)
            {
                gy_pos += gy_coords[d] * gy_strides[d];
            }
            
            gx_data[i] = gy_data[gy_pos];
        }
        
        gxs.push_back(Tensor(gx_data, x_shape, gy.dtype()));
        offset_in_dim += x_shape[dim_];
    }
    
    return gxs;
}

Tensor cat(const std::vector<Tensor> &xs, int dim)
{
    auto op = std::make_shared<Cat>(dim);
    auto outputs = (*op)(xs);
    return outputs[0];
}

}  // namespace origin

