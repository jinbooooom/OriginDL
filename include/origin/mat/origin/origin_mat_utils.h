#ifndef __ORIGIN_DL_ORIGIN_MAT_UTILS_H__
#define __ORIGIN_DL_ORIGIN_MAT_UTILS_H__

#include <string>
#include <vector>
#include "origin/mat/origin/../basic_types.h"

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
                      const std::vector<data_t> &data_vec,
                      const std::vector<size_t> &shape,
                      DataType dtype,
                      const std::string &device_str);

void print_libtorch_style(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape);

void print_simple_format(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape);

void print_slice_format(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape);

void print_slice_recursive(const std::vector<data_t> &data_vec,
                           const std::vector<size_t> &shape,
                           std::vector<size_t> &indices,
                           size_t current_dim,
                           size_t display_dims);

void print_slice_content(const std::vector<data_t> &data_vec,
                         const std::vector<size_t> &shape,
                         const std::vector<size_t> &indices,
                         size_t display_dims);

// 格式化工具函数
std::string format_element(data_t value, DataType dtype);
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

void print_memory_layout(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape);

void print_tensor_stats(const std::vector<data_t> &data_vec, const std::vector<size_t> &shape);

void print_access_pattern(const std::vector<size_t> &shape);
}  // namespace debug

namespace compute
{
// === 计算工具函数 ===
// 用于张量的数学计算和索引操作

size_t calculate_linear_index(const std::vector<size_t> &indices, const std::vector<size_t> &shape);

std::vector<size_t> calculate_indices_from_linear(size_t linear_index, const std::vector<size_t> &shape);

// 统计工具函数
data_t calculate_sum(const std::vector<data_t> &data_vec);
data_t calculate_mean(const std::vector<data_t> &data_vec);
data_t calculate_max(const std::vector<data_t> &data_vec);
data_t calculate_min(const std::vector<data_t> &data_vec);
data_t calculate_std(const std::vector<data_t> &data_vec);

// 转换工具函数
std::vector<data_t> convert_to_vector(const void *data_ptr, size_t elements, DataType dtype);

void convert_from_vector(const std::vector<data_t> &data_vec, void *data_ptr, DataType dtype);
}  // namespace compute

namespace validate
{
// === 验证工具函数 ===
// 用于张量数据的验证和比较

bool validate_shape(const std::vector<size_t> &shape);
bool validate_indices(const std::vector<size_t> &indices, const std::vector<size_t> &shape);

bool compare_tensors(const std::vector<data_t> &data1, const std::vector<data_t> &data2, data_t tolerance = 1e-6);

bool is_same_shape(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2);

bool is_broadcastable(const std::vector<size_t> &shape1, const std::vector<size_t> &shape2);
}  // namespace validate

}  // namespace utils
}  // namespace origin

#endif  // __ORIGIN_DL_ORIGIN_MAT_UTILS_H__
