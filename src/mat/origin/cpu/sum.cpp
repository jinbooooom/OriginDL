#include "origin/mat/origin/origin_mat.h"
#include <stdexcept>

namespace origin {
namespace cpu {

// 前向声明
data_t sum_all(const OriginMat& mat);

std::unique_ptr<OriginMat> sum(const OriginMat& mat, int axis) {
    if (axis == -1) {
        // 对所有元素求和，返回标量
        data_t sum_value = sum_all(mat);
        Shape result_shape = {1};  // 标量结果
        return std::make_unique<OriginMat>(sum_value, result_shape);
    }
    
    // 验证轴的有效性
    if (axis < 0 || axis >= static_cast<int>(mat.shape().size())) {
        throw std::invalid_argument("Invalid axis for sum operation");
    }
    
    // 计算结果形状：移除指定轴
    std::vector<size_t> result_dims;
    for (size_t i = 0; i < mat.shape().size(); ++i) {
        if (i != static_cast<size_t>(axis)) {
            result_dims.push_back(mat.shape()[i]);
        }
    }
    Shape result_shape(result_dims);
    
    auto result = std::make_unique<OriginMat>(result_shape, mat.dtype());
    
    // 执行轴求和
    switch (mat.dtype()) {
        case DataType::kFloat32: {
            const float* src_data = mat.data_ptr<float>();
            float* dst_data = result->data_ptr<float>();
            
            // 计算每个输出位置的索引
            std::vector<size_t> src_indices(mat.shape().size(), 0);
            std::vector<size_t> dst_indices(result_shape.size(), 0);
            
            for (size_t dst_idx = 0; dst_idx < result_shape.elements(); ++dst_idx) {
                // 将一维索引转换为多维索引
                size_t temp = dst_idx;
                for (int i = result_shape.size() - 1; i >= 0; --i) {
                    dst_indices[i] = temp % result_shape[i];
                    temp /= result_shape[i];
                }
                
                // 构建源索引
                size_t src_idx = 0;
                for (size_t i = 0; i < mat.shape().size(); ++i) {
                    if (i == static_cast<size_t>(axis)) {
                        src_indices[i] = 0;  // 轴维度从0开始
                    } else {
                        // 找到对应的输出维度索引
                        size_t output_dim = (i < static_cast<size_t>(axis)) ? i : i - 1;
                        src_indices[i] = dst_indices[output_dim];
                    }
                }
                
                // 计算线性索引并求和
                float sum_val = 0.0f;
                for (size_t axis_val = 0; axis_val < mat.shape()[axis]; ++axis_val) {
                    src_indices[axis] = axis_val;
                    
                    // 计算源线性索引
                    size_t src_linear_idx = 0;
                    size_t stride = 1;
                    for (int i = mat.shape().size() - 1; i >= 0; --i) {
                        src_linear_idx += src_indices[i] * stride;
                        stride *= mat.shape()[i];
                    }
                    
                    sum_val += src_data[src_linear_idx];
                }
                
                dst_data[dst_idx] = sum_val;
            }
            break;
        }
        case DataType::kFloat64: {
            const double* src_data = mat.data_ptr<double>();
            double* dst_data = result->data_ptr<double>();
            
            // 计算每个输出位置的索引
            std::vector<size_t> src_indices(mat.shape().size(), 0);
            std::vector<size_t> dst_indices(result_shape.size(), 0);
            
            for (size_t dst_idx = 0; dst_idx < result_shape.elements(); ++dst_idx) {
                // 将一维索引转换为多维索引
                size_t temp = dst_idx;
                for (int i = result_shape.size() - 1; i >= 0; --i) {
                    dst_indices[i] = temp % result_shape[i];
                    temp /= result_shape[i];
                }
                
                // 构建源索引
                size_t src_idx = 0;
                for (size_t i = 0; i < mat.shape().size(); ++i) {
                    if (i == static_cast<size_t>(axis)) {
                        src_indices[i] = 0;  // 轴维度从0开始
                    } else {
                        // 找到对应的输出维度索引
                        size_t output_dim = (i < static_cast<size_t>(axis)) ? i : i - 1;
                        src_indices[i] = dst_indices[output_dim];
                    }
                }
                
                // 计算线性索引并求和
                double sum_val = 0.0;
                for (size_t axis_val = 0; axis_val < mat.shape()[axis]; ++axis_val) {
                    src_indices[axis] = axis_val;
                    
                    // 计算源线性索引
                    size_t src_linear_idx = 0;
                    size_t stride = 1;
                    for (int i = mat.shape().size() - 1; i >= 0; --i) {
                        src_linear_idx += src_indices[i] * stride;
                        stride *= mat.shape()[i];
                    }
                    
                    sum_val += src_data[src_linear_idx];
                }
                
                dst_data[dst_idx] = sum_val;
            }
            break;
        }
        case DataType::kInt32: {
            const int32_t* src_data = mat.data_ptr<int32_t>();
            int32_t* dst_data = result->data_ptr<int32_t>();
            
            // 计算每个输出位置的索引
            std::vector<size_t> src_indices(mat.shape().size(), 0);
            std::vector<size_t> dst_indices(result_shape.size(), 0);
            
            for (size_t dst_idx = 0; dst_idx < result_shape.elements(); ++dst_idx) {
                // 将一维索引转换为多维索引
                size_t temp = dst_idx;
                for (int i = result_shape.size() - 1; i >= 0; --i) {
                    dst_indices[i] = temp % result_shape[i];
                    temp /= result_shape[i];
                }
                
                // 构建源索引
                size_t src_idx = 0;
                for (size_t i = 0; i < mat.shape().size(); ++i) {
                    if (i == static_cast<size_t>(axis)) {
                        src_indices[i] = 0;  // 轴维度从0开始
                    } else {
                        // 找到对应的输出维度索引
                        size_t output_dim = (i < static_cast<size_t>(axis)) ? i : i - 1;
                        src_indices[i] = dst_indices[output_dim];
                    }
                }
                
                // 计算线性索引并求和
                int32_t sum_val = 0;
                for (size_t axis_val = 0; axis_val < mat.shape()[axis]; ++axis_val) {
                    src_indices[axis] = axis_val;
                    
                    // 计算源线性索引
                    size_t src_linear_idx = 0;
                    size_t stride = 1;
                    for (int i = mat.shape().size() - 1; i >= 0; --i) {
                        src_linear_idx += src_indices[i] * stride;
                        stride *= mat.shape()[i];
                    }
                    
                    sum_val += src_data[src_linear_idx];
                }
                
                dst_data[dst_idx] = sum_val;
            }
            break;
        }
        case DataType::kInt8: {
            const int8_t* src_data = mat.data_ptr<int8_t>();
            int8_t* dst_data = result->data_ptr<int8_t>();
            
            // 计算每个输出位置的索引
            std::vector<size_t> src_indices(mat.shape().size(), 0);
            std::vector<size_t> dst_indices(result_shape.size(), 0);
            
            for (size_t dst_idx = 0; dst_idx < result_shape.elements(); ++dst_idx) {
                // 将一维索引转换为多维索引
                size_t temp = dst_idx;
                for (int i = result_shape.size() - 1; i >= 0; --i) {
                    dst_indices[i] = temp % result_shape[i];
                    temp /= result_shape[i];
                }
                
                // 构建源索引
                size_t src_idx = 0;
                for (size_t i = 0; i < mat.shape().size(); ++i) {
                    if (i == static_cast<size_t>(axis)) {
                        src_indices[i] = 0;  // 轴维度从0开始
                    } else {
                        // 找到对应的输出维度索引
                        size_t output_dim = (i < static_cast<size_t>(axis)) ? i : i - 1;
                        src_indices[i] = dst_indices[output_dim];
                    }
                }
                
                // 计算线性索引并求和
                int8_t sum_val = 0;
                for (size_t axis_val = 0; axis_val < mat.shape()[axis]; ++axis_val) {
                    src_indices[axis] = axis_val;
                    
                    // 计算源线性索引
                    size_t src_linear_idx = 0;
                    size_t stride = 1;
                    for (int i = mat.shape().size() - 1; i >= 0; --i) {
                        src_linear_idx += src_indices[i] * stride;
                        stride *= mat.shape()[i];
                    }
                    
                    sum_val += src_data[src_linear_idx];
                }
                
                dst_data[dst_idx] = sum_val;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for sum operation");
    }

    return result;
}

data_t sum_all(const OriginMat& mat) {
    switch (mat.dtype()) {
        case DataType::kFloat32: {
            const float* data = mat.data_ptr<float>();
            float sum = 0.0f;
            for (size_t i = 0; i < mat.elements(); ++i) {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        case DataType::kFloat64: {
            const double* data = mat.data_ptr<double>();
            double sum = 0.0;
            for (size_t i = 0; i < mat.elements(); ++i) {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        case DataType::kInt32: {
            const int32_t* data = mat.data_ptr<int32_t>();
            int32_t sum = 0;
            for (size_t i = 0; i < mat.elements(); ++i) {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        case DataType::kInt8: {
            const int8_t* data = mat.data_ptr<int8_t>();
            int8_t sum = 0;
            for (size_t i = 0; i < mat.elements(); ++i) {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        default:
            throw std::invalid_argument("Unsupported data type for sum_all");
    }
}

} // namespace cpu
} // namespace origin
