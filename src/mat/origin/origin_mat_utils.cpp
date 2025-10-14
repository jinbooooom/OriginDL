#include "origin/mat/origin/origin_mat_utils.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace origin
{
namespace utils
{

// === 显示配置常量定义 ===
// 1D张量显示配置
const size_t kMax1DElements = 20;
// 2D张量显示配置
const size_t kMax2DRows = 10;
const size_t kMax2DCols = 20;
// 高维张量切片显示配置
const size_t kMaxSliceRows = 10;
const size_t kMaxSliceCols = 10;
// 高维切片剩余元素显示配置
const size_t kMaxRemainingElements = 5;

// === OriginMat工具函数实现 ===

namespace visualize
{

void print_origin_mat(const std::string &desc,
                      const std::vector<data_t> &data_vec,
                      const std::vector<size_t> &shape,
                      DataType dtype,
                      const std::string &device_str)
{
    std::cout << desc << ":\n";

    if (data_vec.empty())
    {
        std::cout << "(null)" << std::endl;
        return;
    }

    if (shape.empty())
    {
        // 标量
        std::cout << "(" << format_element(data_vec[0], dtype) << ")" << std::endl;
        return;
    }

    // 使用LibTorch风格的打印
    print_libtorch_style(data_vec, shape);
    std::cout << std::endl;

    // 基本信息最后打印
    std::cout << " OriginMat(shape=" << format_shape(shape) << ", dtype=" << format_dtype(dtype)
              << ", device=" << format_device(device_str) << ")" << std::endl;
}

void print_libtorch_style(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape)
{
    if (shape.size() <= 2)
    {
        // 对于1D和2D张量，使用简单的格式
        print_simple_format(data_vec, shape);
        return;
    }

    // 对于3D及以上张量，使用LibTorch的切片格式
    print_slice_format(data_vec, shape);
}

void print_simple_format(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape)
{
    if (shape.size() == 1)
    {
        // 1D张量
        std::cout << "[";
        size_t max_show = std::min(shape[0], kMax1DElements);
        for (size_t i = 0; i < max_show; ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << format_element(data_vec[i], DataType::kFloat32);
        }
        if (shape[0] > max_show)
        {
            std::cout << ", ...";
        }
        std::cout << "]";
    }
    else if (shape.size() == 2)
    {
        // 2D张量
        std::cout << "[";
        size_t max_rows = std::min(shape[0], kMax2DRows);
        size_t max_cols = std::min(shape[1], kMax2DCols);

        for (size_t i = 0; i < max_rows; ++i)
        {
            if (i > 0)
            {
                std::cout << "," << std::endl << " ";
            }
            std::cout << "[";
            for (size_t j = 0; j < max_cols; ++j)
            {
                if (j > 0)
                    std::cout << ", ";
                size_t index = i * shape[1] + j;
                std::cout << format_element(data_vec[index], DataType::kFloat32);
            }
            if (shape[1] > max_cols)
            {
                std::cout << ", ...";
            }
            std::cout << "]";
        }
        if (shape[0] > max_rows)
        {
            std::cout << "," << std::endl << " ...";
        }
        std::cout << "]";
    }
}

void print_slice_format(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape)
{
    if (shape.size() <= 2)
    {
        // 对于1D和2D张量，直接使用简单格式
        print_simple_format(data_vec, shape);
        return;
    }

    // 对于3D及以上张量，确定显示维度数
    size_t display_dims;
    if (shape.size() == 3)
    {
        display_dims = 1;  // 3D: 显示 (i,.,.) 切片
    }
    else if (shape.size() == 4)
    {
        display_dims = 2;  // 4D: 显示 (i,j,.,.) 切片
    }
    else
    {
        display_dims = 3;  // 5D+: 显示 (i,j,k,.,.) 切片
    }

    // 递归遍历显示维度
    std::vector<size_t> indices(display_dims, 0);
    print_slice_recursive(data_vec, shape, indices, 0, display_dims);
}

void print_slice_recursive(const std::vector<data_t> &data_vec,
                           const std::vector<size_t> &shape,
                           std::vector<size_t> &indices,
                           size_t current_dim,
                           size_t display_dims)
{
    if (current_dim == display_dims)
    {
        // 打印索引标签
        std::cout << "(";
        for (size_t i = 0; i < indices.size(); ++i)
        {
            if (i > 0)
                std::cout << ",";
            std::cout << indices[i];
        }
        for (size_t i = display_dims; i < shape.size(); ++i)
        {
            std::cout << ",.";
        }
        std::cout << ") = " << std::endl;

        // 打印切片内容
        print_slice_content(data_vec, shape, indices, display_dims);
        std::cout << std::endl;
        return;
    }

    // 递归遍历当前维度
    for (size_t i = 0; i < shape[current_dim]; ++i)
    {
        indices[current_dim] = i;
        print_slice_recursive(data_vec, shape, indices, current_dim + 1, display_dims);
    }
}

void print_slice_content(const std::vector<data_t> &data_vec,
                         const std::vector<size_t> &shape,
                         const std::vector<size_t> &indices,
                         size_t display_dims)
{
    // 计算当前切片的起始索引
    size_t start_index = 0;
    size_t multiplier  = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        if (static_cast<size_t>(i) < display_dims)
        {
            start_index += indices[i] * multiplier;
        }
        multiplier *= shape[i];
    }

    // 计算切片形状（去掉前display_dims个维度）
    std::vector<size_t> slice_shape(shape.begin() + display_dims, shape.end());

    if (slice_shape.empty())
    {
        // 没有剩余维度，直接打印单个元素
        std::cout << std::setw(6) << format_element(data_vec[start_index], DataType::kFloat32);
    }
    else if (slice_shape.size() == 1)
    {
        // 1D切片
        for (size_t i = 0; i < slice_shape[0]; ++i)
        {
            if (i > 0)
                std::cout << "  ";
            std::cout << std::setw(6) << format_element(data_vec[start_index + i], DataType::kFloat32);
        }
    }
    else if (slice_shape.size() == 2)
    {
        // 2D切片 - 这是最常见的情况，添加省略功能
        size_t max_rows = std::min(slice_shape[0], kMaxSliceRows);
        size_t max_cols = std::min(slice_shape[1], kMaxSliceCols);

        for (size_t i = 0; i < max_rows; ++i)
        {
            if (i > 0)
                std::cout << std::endl;
            for (size_t j = 0; j < max_cols; ++j)
            {
                if (j > 0)
                    std::cout << "  ";
                size_t index = start_index + i * slice_shape[1] + j;
                std::cout << std::setw(6) << format_element(data_vec[index], DataType::kFloat32);
            }
            if (slice_shape[1] > max_cols)
            {
                std::cout << "  ...";
            }
        }
        if (slice_shape[0] > max_rows)
        {
            std::cout << std::endl << " ...";
        }
    }
    else
    {
        // 3D及以上切片，显示为2D矩阵（取前两个维度）
        size_t dim0               = slice_shape[0];
        size_t dim1               = slice_shape[1];
        size_t remaining_elements = 1;
        for (size_t k = 2; k < slice_shape.size(); ++k)
        {
            remaining_elements *= slice_shape[k];
        }

        for (size_t i = 0; i < dim0; ++i)
        {
            if (i > 0)
                std::cout << std::endl;
            for (size_t j = 0; j < dim1; ++j)
            {
                if (j > 0)
                    std::cout << "  ";
                // 计算在剩余维度中的索引
                size_t base_index = start_index + i * dim1 * remaining_elements + j * remaining_elements;

                // 显示剩余维度的第一个元素（简化显示）
                if (remaining_elements == 1)
                {
                    std::cout << std::setw(6) << format_element(data_vec[base_index], DataType::kFloat32);
                }
                else
                {
                    // 对于多个剩余元素，显示前几个
                    size_t show_elements = std::min(remaining_elements, kMaxRemainingElements);
                    std::cout << "[";
                    for (size_t k = 0; k < show_elements; ++k)
                    {
                        if (k > 0)
                            std::cout << ",";
                        std::cout << format_element(data_vec[base_index + k], DataType::kFloat32);
                    }
                    if (remaining_elements > show_elements)
                    {
                        std::cout << ",...";
                    }
                    std::cout << "]";
                }
            }
        }
    }
}

std::string format_element(data_t value, DataType dtype)
{
    std::ostringstream oss;

    // 根据数据类型格式化
    switch (dtype)
    {
        case DataType::kFloat32:
        case DataType::kFloat64:
            if (value == static_cast<int>(value) && value >= -1000 && value <= 1000)
            {
                // 整数显示为整数
                oss << static_cast<int>(value);
            }
            else
            {
                // 浮点数显示
                oss << std::fixed << std::setprecision(1) << value;
            }
            break;
        case DataType::kInt32:
        case DataType::kInt8:
            oss << static_cast<int>(value);
            break;
        default:
            oss << value;
            break;
    }

    return oss.str();
}

std::string format_shape(const std::vector<size_t> &shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (i > 0)
            oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

std::string format_dtype(DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return "float32";
        case DataType::kFloat64:
            return "float64";
        case DataType::kInt32:
            return "int32";
        case DataType::kInt8:
            return "int8";
        default:
            return "unknown";
    }
}

std::string format_device(const std::string &device_str)
{
    return device_str;  // 直接返回，可以在这里添加格式化逻辑
}

}  // namespace visualize

namespace debug
{

void print_debug_info(const std::string &desc,
                      const std::vector<size_t> &shape,
                      DataType dtype,
                      const std::string &device_str)
{
    std::cout << "=== Debug Info: " << desc << " ===" << std::endl;
    std::cout << "Shape: " << visualize::format_shape(shape) << std::endl;
    std::cout << "Dtype: " << visualize::format_dtype(dtype) << std::endl;
    std::cout << "Device: " << visualize::format_device(device_str) << std::endl;
    std::cout << "Elements: "
              << (shape.empty() ? 0 : std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>()))
              << std::endl;
    std::cout << "================================" << std::endl;
}

void print_memory_layout(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape)
{
    std::cout << "=== Memory Layout ===" << std::endl;
    for (size_t i = 0; i < std::min(data_vec.size(), size_t(20)); ++i)
    {
        std::cout << "[" << i << "] = " << data_vec[i] << std::endl;
    }
    if (data_vec.size() > 20)
    {
        std::cout << "... (showing first 20 elements)" << std::endl;
    }
    std::cout << "====================" << std::endl;
}

void print_tensor_stats(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape)
{
    if (data_vec.empty())
        return;

    std::cout << "=== Tensor Statistics ===" << std::endl;
    std::cout << "Sum: " << compute::calculate_sum(data_vec) << std::endl;
    std::cout << "Mean: " << compute::calculate_mean(data_vec) << std::endl;
    std::cout << "Max: " << compute::calculate_max(data_vec) << std::endl;
    std::cout << "Min: " << compute::calculate_min(data_vec) << std::endl;
    std::cout << "Std: " << compute::calculate_std(data_vec) << std::endl;
    std::cout << "=========================" << std::endl;
}

void print_access_pattern(const std::vector<size_t> &shape)
{
    std::cout << "=== Access Pattern ===" << std::endl;
    std::cout << "Shape: " << visualize::format_shape(shape) << std::endl;
    std::cout << "LibTorch style access order:" << std::endl;

    if (shape.size() >= 2)
    {
        for (size_t i = 0; i < shape[0]; ++i)
        {
            for (size_t j = 0; j < shape[1]; ++j)
            {
                std::cout << "(" << i << "," << j;
                for (size_t k = 2; k < shape.size(); ++k)
                {
                    std::cout << ",.";
                }
                std::cout << ") ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << "=====================" << std::endl;
}

}  // namespace debug

namespace compute
{

size_t calculate_linear_index(const std::vector<size_t> &indices, const std::vector<size_t> &shape)
{
    size_t linear_index = 0;
    size_t multiplier   = 1;
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i)
    {
        linear_index += indices[i] * multiplier;
        multiplier *= shape[i];
    }
    return linear_index;
}

std::vector<size_t> calculate_indices_from_linear(size_t linear_index, const std::vector<size_t> &shape)
{
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i)
    {
        indices[i] = linear_index % shape[i];
        linear_index /= shape[i];
    }
    return indices;
}

data_t calculate_sum(const std::vector<data_t> &data_vec)
{
    data_t sum = 0;
    for (data_t value : data_vec)
    {
        sum += value;
    }
    return sum;
}

data_t calculate_mean(const std::vector<data_t> &data_vec)
{
    if (data_vec.empty())
        return 0;
    return calculate_sum(data_vec) / static_cast<data_t>(data_vec.size());
}

data_t calculate_max(const std::vector<data_t> &data_vec)
{
    if (data_vec.empty())
        return 0;
    return *std::max_element(data_vec.begin(), data_vec.end());
}

data_t calculate_min(const std::vector<data_t> &data_vec)
{
    if (data_vec.empty())
        return 0;
    return *std::min_element(data_vec.begin(), data_vec.end());
}

data_t calculate_std(const std::vector<data_t> &data_vec)
{
    if (data_vec.empty())
        return 0;

    data_t mean     = calculate_mean(data_vec);
    data_t variance = 0;
    for (data_t value : data_vec)
    {
        variance += (value - mean) * (value - mean);
    }
    variance /= static_cast<data_t>(data_vec.size());
    return std::sqrt(variance);
}

std::vector<data_t> convert_to_vector(const void *data_ptr, size_t elements, DataType dtype)
{
    std::vector<data_t> result(elements);

    switch (dtype)
    {
        case DataType::kFloat32:
        {
            const float *data = static_cast<const float *>(data_ptr);
            for (size_t i = 0; i < elements; ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *data = static_cast<const double *>(data_ptr);
            for (size_t i = 0; i < elements; ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *data = static_cast<const int32_t *>(data_ptr);
            for (size_t i = 0; i < elements; ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *data = static_cast<const int8_t *>(data_ptr);
            for (size_t i = 0; i < elements; ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for conversion");
    }

    return result;
}

void convert_from_vector(const std::vector<data_t> &data_vec, void *data_ptr, DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
        {
            float *data = static_cast<float *>(data_ptr);
            for (size_t i = 0; i < data_vec.size(); ++i)
            {
                data[i] = static_cast<float>(data_vec[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = static_cast<double *>(data_ptr);
            for (size_t i = 0; i < data_vec.size(); ++i)
            {
                data[i] = static_cast<double>(data_vec[i]);
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = static_cast<int32_t *>(data_ptr);
            for (size_t i = 0; i < data_vec.size(); ++i)
            {
                data[i] = static_cast<int32_t>(data_vec[i]);
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = static_cast<int8_t *>(data_ptr);
            for (size_t i = 0; i < data_vec.size(); ++i)
            {
                data[i] = static_cast<int8_t>(data_vec[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for conversion");
    }
}

}  // namespace compute

namespace validate
{

bool validate_shape(const std::vector<size_t> &shape)
{
    if (shape.empty())
        return false;
    for (size_t dim : shape)
    {
        if (dim == 0)
            return false;
    }
    return true;
}

bool validate_indices(const std::vector<size_t> &indices, const std::vector<size_t> &shape)
{
    if (indices.size() != shape.size())
        return false;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        if (indices[i] >= shape[i])
            return false;
    }
    return true;
}

bool compare_tensors(const std::vector<data_t> &data1, const std::vector<data_t> &data2, data_t tolerance)
{
    if (data1.size() != data2.size())
        return false;
    for (size_t i = 0; i < data1.size(); ++i)
    {
        if (std::abs(data1[i] - data2[i]) > tolerance)
            return false;
    }
    return true;
}

bool is_same_shape(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2)
{
    if (shape1.size() != shape2.size())
        return false;
    for (size_t i = 0; i < shape1.size(); ++i)
    {
        if (shape1[i] != shape2[i])
            return false;
    }
    return true;
}

bool is_broadcastable(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2)
{
    // 简化的广播检查实现
    size_t min_size = std::min(shape1.size(), shape2.size());
    for (size_t i = 0; i < min_size; ++i)
    {
        size_t dim1 = shape1[shape1.size() - 1 - i];
        size_t dim2 = shape2[shape2.size() - 1 - i];
        if (dim1 != dim2 && dim1 != 1 && dim2 != 1)
        {
            return false;
        }
    }
    return true;
}

}  // namespace validate

}  // namespace utils
}  // namespace origin