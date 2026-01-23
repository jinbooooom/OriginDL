#ifndef __ORIGIN_DL_ORIGIN_MAT_UTILS_H__
#define __ORIGIN_DL_ORIGIN_MAT_UTILS_H__

#include <cmath>
#include <string>
#include <vector>
#include "origin/mat/origin/../basic_types.h"
#include "origin/mat/origin/../shape.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace utils
{

// === 显示配置常量 ===
namespace config
{
// 1D张量显示配置
extern const size_t kMax1DElements;

// 2D张量显示配置
extern const size_t kMax2DRows;
extern const size_t kMax2DCols;

// 高维张量切片显示配置
extern const size_t kMaxSliceRows;
extern const size_t kMaxSliceCols;

// 高维切片剩余元素显示配置
extern const size_t kMaxRemainingElements;
}  // namespace config

// === OriginMat工具函数命名空间 ===
// 按功能分类到不同的子命名空间，提供更清晰的API

namespace visualize
{
// === 可视化工具函数 ===
// 用于张量的显示和格式化

void print_origin_mat(const std::string &desc,
                      const std::vector<float> &data_vec,
                      const std::vector<size_t> &shape,
                      DataType dtype,
                      const std::string &device_str);

void print_libtorch_style(const std::vector<float> &data_vec, const std::vector<size_t> &shape);

void print_simple_format(const std::vector<float> &data_vec, const std::vector<size_t> &shape);

void print_slice_format(const std::vector<float> &data_vec, const std::vector<size_t> &shape);

void print_slice_recursive(const std::vector<float> &data_vec,
                           const std::vector<size_t> &shape,
                           std::vector<size_t> &indices,
                           size_t current_dim,
                           size_t display_dims);

void print_slice_content(const std::vector<float> &data_vec,
                         const std::vector<size_t> &shape,
                         const std::vector<size_t> &indices,
                         size_t display_dims);

// 格式化工具函数
std::string format_element(float value, DataType dtype);
std::string format_shape(const std::vector<size_t> &shape);
std::string format_dtype(DataType dtype);
std::string format_device(const std::string &device_str);
}  // namespace visualize

namespace debug
{
// === 调试工具函数 ===
// 用于开发和调试过程中的信息输出

void print_debug_info(const std::string &desc,
                      const std::vector<size_t> &shape,
                      DataType dtype,
                      const std::string &device_str);

void print_memory_layout(const std::vector<float> &data_vec, const std::vector<size_t> &shape);

void print_tensor_stats(const std::vector<float> &data_vec, const std::vector<size_t> &shape);

void print_access_pattern(const std::vector<size_t> &shape);
}  // namespace debug

namespace compute
{
// === 计算工具函数 ===
// 用于张量的数学计算和索引操作

size_t calculate_linear_index(const std::vector<size_t> &indices, const std::vector<size_t> &shape);

std::vector<size_t> calculate_indices_from_linear(size_t linear_index, const std::vector<size_t> &shape);

// 统计工具函数
float calculate_sum(const std::vector<float> &data_vec);
float calculate_mean(const std::vector<float> &data_vec);
float calculate_max(const std::vector<float> &data_vec);
float calculate_min(const std::vector<float> &data_vec);
float calculate_std(const std::vector<float> &data_vec);

// 转换工具函数
std::vector<float> convert_to_vector(const void *data_ptr, size_t elements, DataType dtype);

void convert_from_vector(const std::vector<float> &data_vec, void *data_ptr, DataType dtype);

// 标量值提取函数
Scalar get_scalar_value(const void *data_ptr, DataType dtype);

// 广播形状计算函数
Shape compute_broadcast_shape(const OriginMat &a, const OriginMat &b);
}  // namespace compute

namespace validate
{
// === 验证工具函数 ===
// 用于张量数据的验证和比较

bool validate_shape(const std::vector<size_t> &shape);
bool validate_indices(const std::vector<size_t> &indices, const std::vector<size_t> &shape);

bool compare_tensors(const std::vector<float> &data1, const std::vector<float> &data2, float tolerance = 1e-6);

bool is_same_shape(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2);

bool is_broadcastable(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2);

// === 设备验证宏 ===
// 验证两个张量在同一设备上
#define VALIDATE_SAME_DEVICE(a, b)                                                                           \
    do                                                                                                       \
    {                                                                                                        \
        if (unlikely((a).device() != (b).device()))                                                          \
        {                                                                                                    \
            THROW_INVALID_ARG(                                                                               \
                "Expected all tensors to be on the same device, but found at least two devices, {} and {}!", \
                (a).device().to_string(), (b).device().to_string());                                         \
        }                                                                                                    \
    } while (0)

// 验证张量在指定设备上
#define VALIDATE_DEVICE(tensor, expected_device)                                                 \
    do                                                                                           \
    {                                                                                            \
        if (unlikely((tensor).device().type() != (expected_device)))                             \
        {                                                                                        \
            std::string expected_str = ((expected_device) == DeviceType::kCPU) ? "cpu" : "cuda"; \
            THROW_INVALID_ARG("Expected tensor to be on {} device, but got {}!", expected_str,   \
                              (tensor).device().to_string());                                    \
        }                                                                                        \
    } while (0)

// 验证张量在CUDA设备上
#define VALIDATE_CUDA_DEVICE(tensor)                                                                               \
    do                                                                                                             \
    {                                                                                                              \
        if (unlikely((tensor).device().type() != DeviceType::kCUDA))                                               \
        {                                                                                                          \
            THROW_INVALID_ARG("Expected tensor to be on cuda device, but got {}!", (tensor).device().to_string()); \
        }                                                                                                          \
    } while (0)

// 验证张量在CPU设备上
#define VALIDATE_CPU_DEVICE(tensor)                                                                               \
    do                                                                                                            \
    {                                                                                                             \
        if (unlikely((tensor).device().type() != DeviceType::kCPU))                                               \
        {                                                                                                         \
            THROW_INVALID_ARG("Expected tensor to be on cpu device, but got {}!", (tensor).device().to_string()); \
        }                                                                                                         \
    } while (0)

// 验证两个张量在同一CUDA设备上
#define VALIDATE_SAME_CUDA_DEVICE(a, b)                                                                       \
    do                                                                                                        \
    {                                                                                                         \
        if (unlikely((a).device() != (b).device()))                                                           \
        {                                                                                                     \
            THROW_INVALID_ARG(                                                                                \
                "Expected all tensors to be on the same device, but found at least two devices, {} and {}!",  \
                (a).device().to_string(), (b).device().to_string());                                          \
        }                                                                                                     \
        if (unlikely((a).device().type() != DeviceType::kCUDA))                                               \
        {                                                                                                     \
            THROW_INVALID_ARG("Expected tensor to be on cuda device, but got {}!", (a).device().to_string()); \
        }                                                                                                     \
    } while (0)

// 验证两个张量在同一CPU设备上
#define VALIDATE_SAME_CPU_DEVICE(a, b)                                                                       \
    do                                                                                                       \
    {                                                                                                        \
        if (unlikely((a).device() != (b).device()))                                                          \
        {                                                                                                    \
            THROW_INVALID_ARG(                                                                               \
                "Expected all tensors to be on the same device, but found at least two devices, {} and {}!", \
                (a).device().to_string(), (b).device().to_string());                                         \
        }                                                                                                    \
        if (unlikely((a).device().type() != DeviceType::kCPU))                                               \
        {                                                                                                    \
            THROW_INVALID_ARG("Expected tensor to be on cpu device, but got {}!", (a).device().to_string()); \
        }                                                                                                    \
    } while (0)

// === 数据类型验证宏 ===
// 验证两个张量的数据类型是否相同
#define VALIDATE_SAME_DTYPE(a, b)                                                            \
    do                                                                                       \
    {                                                                                        \
        if (unlikely((a).dtype() != (b).dtype()))                                            \
        {                                                                                    \
            THROW_INVALID_ARG("Data type mismatch: {} vs {}!", dtype_to_string((a).dtype()), \
                              dtype_to_string((b).dtype()));                                 \
        }                                                                                    \
    } while (0)

// 验证张量的数据类型是否是指定类型
#define VALIDATE_DTYPE(tensor, expected_dtype)                                                                  \
    do                                                                                                          \
    {                                                                                                           \
        if (unlikely((tensor).dtype() != (expected_dtype)))                                                     \
        {                                                                                                       \
            THROW_INVALID_ARG("Expected tensor to be of type {}, but got {}!", dtype_to_string(expected_dtype), \
                              dtype_to_string((tensor).dtype()));                                               \
        }                                                                                                       \
    } while (0)

// 验证张量是否支持浮点运算（用于exp, log, sqrt等数学函数）
#define VALIDATE_FLOAT_DTYPE(tensor)                                                                              \
    do                                                                                                            \
    {                                                                                                             \
        if (unlikely((tensor).dtype() != DataType::kFloat32 && (tensor).dtype() != DataType::kFloat64))           \
        {                                                                                                         \
            THROW_INVALID_ARG("Mathematical operation only supported for float32 and float64 types, but got {}!", \
                              dtype_to_string((tensor).dtype()));                                                 \
        }                                                                                                         \
    } while (0)

// 验证两个张量是否都支持浮点运算
#define VALIDATE_FLOAT_DTYPE_2(a, b)                                                                              \
    do                                                                                                            \
    {                                                                                                             \
        if (unlikely((a).dtype() != DataType::kFloat32 && (a).dtype() != DataType::kFloat64))                     \
        {                                                                                                         \
            THROW_INVALID_ARG("Mathematical operation only supported for float32 and float64 types, but got {}!", \
                              dtype_to_string((a).dtype()));                                                      \
        }                                                                                                         \
        if (unlikely((b).dtype() != DataType::kFloat32 && (b).dtype() != DataType::kFloat64))                     \
        {                                                                                                         \
            THROW_INVALID_ARG("Mathematical operation only supported for float32 and float64 types, but got {}!", \
                              dtype_to_string((b).dtype()));                                                      \
        }                                                                                                         \
    } while (0)

}  // namespace validate

// === OriginMat工具函数 ===
// 从OriginMat类中提取的工具函数

void validate_shape(const Shape &shape);
std::vector<size_t> compute_strides(const Shape &shape);

bool can_broadcast_to(const Shape &source_shape, const Shape &target_shape);

}  // namespace utils
}  // namespace origin

#endif  // __ORIGIN_DL_ORIGIN_MAT_UTILS_H__
