#ifndef __ORIGIN_MAT_DEVICE_VALIDATION_H__
#define __ORIGIN_MAT_DEVICE_VALIDATION_H__

#include "origin/mat/basic_types.h"
#include "origin/mat/mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace validation
{

/**
 * @brief 验证两个张量是否在同一设备上
 * @param a 第一个张量
 * @param b 第二个张量
 * @param operation_name 操作名称，用于错误信息
 * @throws std::invalid_argument 如果设备不匹配
 */
void validate_same_device(const Mat &a, const Mat &b, const std::string &operation_name);

/**
 * @brief 验证张量是否在指定设备上
 * @param tensor 要验证的张量
 * @param expected_device 期望的设备
 * @param operation_name 操作名称，用于错误信息
 * @throws std::invalid_argument 如果设备不匹配
 */
void validate_device(const Mat &tensor, DeviceType expected_device, const std::string &operation_name);

/**
 * @brief 验证张量是否在CUDA设备上
 * @param tensor 要验证的张量
 * @param operation_name 操作名称，用于错误信息
 * @throws std::invalid_argument 如果不是CUDA设备
 */
void validate_cuda_device(const Mat &tensor, const std::string &operation_name);

/**
 * @brief 验证张量是否在CPU设备上
 * @param tensor 要验证的张量
 * @param operation_name 操作名称，用于错误信息
 * @throws std::invalid_argument 如果不是CPU设备
 */
void validate_cpu_device(const Mat &tensor, const std::string &operation_name);

/**
 * @brief 高效验证两个张量在同一CUDA设备上（组合验证）
 * @param a 第一个张量
 * @param b 第二个张量
 * @param operation_name 操作名称，用于错误信息
 * @throws std::invalid_argument 如果设备不匹配或不是CUDA设备
 */
void validate_same_cuda_device(const Mat &a, const Mat &b, const std::string &operation_name);

/**
 * @brief 高效验证两个张量在同一CPU设备上（组合验证）
 * @param a 第一个张量
 * @param b 第二个张量
 * @param operation_name 操作名称，用于错误信息
 * @throws std::invalid_argument 如果设备不匹配或不是CPU设备
 */
void validate_same_cpu_device(const Mat &a, const Mat &b, const std::string &operation_name);

/**
 * @brief 生成设备不匹配的错误信息
 * @param device1 第一个设备
 * @param device2 第二个设备
 * @param operation_name 操作名称
 * @return 格式化的错误信息
 */
std::string format_device_mismatch_error(const Device &device1, const Device &device2, const std::string &operation_name);

/**
 * @brief 生成设备类型不匹配的错误信息
 * @param actual_device 实际设备
 * @param expected_device 期望设备类型
 * @param operation_name 操作名称
 * @return 格式化的错误信息
 */
std::string format_device_type_mismatch_error(const Device &actual_device, DeviceType expected_device, const std::string &operation_name);

}  // namespace validation
}  // namespace origin

#endif  // __ORIGIN_MAT_DEVICE_VALIDATION_H__
