#include "origin/mat/basic_types.h"
#include "origin/mat/mat.h"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace validation
{

void validate_same_device(const Mat &a, const Mat &b, const std::string &operation_name)
{
    Device device_a = a.device();
    Device device_b = b.device();

    if (device_a != device_b)
    {
        std::string error_msg = format_device_mismatch_error(device_a, device_b, operation_name);
        THROW_INVALID_ARG(error_msg);
    }
}

void validate_device(const Mat &tensor, DeviceType expected_device, const std::string &operation_name)
{
    Device actual_device = tensor.device();

    if (actual_device.type() != expected_device)
    {
        std::string error_msg = format_device_type_mismatch_error(actual_device, expected_device, operation_name);
        THROW_INVALID_ARG(error_msg);
    }
}

void validate_cuda_device(const Mat &tensor, const std::string &operation_name)
{
    validate_device(tensor, DeviceType::kCUDA, operation_name);
}

void validate_cpu_device(const Mat &tensor, const std::string &operation_name)
{
    validate_device(tensor, DeviceType::kCPU, operation_name);
}

void validate_same_cuda_device(const Mat &a, const Mat &b, const std::string &operation_name)
{
    Device device_a = a.device();
    Device device_b = b.device();

    // 首先检查设备类型
    if (device_a.type() != DeviceType::kCUDA)
    {
        std::string error_msg = format_device_type_mismatch_error(device_a, DeviceType::kCUDA, operation_name);
        THROW_INVALID_ARG(error_msg);
    }

    if (device_b.type() != DeviceType::kCUDA)
    {
        std::string error_msg = format_device_type_mismatch_error(device_b, DeviceType::kCUDA, operation_name);
        THROW_INVALID_ARG(error_msg);
    }

    // 然后检查设备是否相同
    if (device_a != device_b)
    {
        std::string error_msg = format_device_mismatch_error(device_a, device_b, operation_name);
        THROW_INVALID_ARG(error_msg);
    }
}

void validate_same_cpu_device(const Mat &a, const Mat &b, const std::string &operation_name)
{
    Device device_a = a.device();
    Device device_b = b.device();

    // 首先检查设备类型
    if (device_a.type() != DeviceType::kCPU)
    {
        std::string error_msg = format_device_type_mismatch_error(device_a, DeviceType::kCPU, operation_name);
        THROW_INVALID_ARG(error_msg);
    }

    if (device_b.type() != DeviceType::kCPU)
    {
        std::string error_msg = format_device_type_mismatch_error(device_b, DeviceType::kCPU, operation_name);
        THROW_INVALID_ARG(error_msg);
    }

    // CPU设备通常不需要检查设备索引，但为了完整性还是检查
    if (device_a != device_b)
    {
        std::string error_msg = format_device_mismatch_error(device_a, device_b, operation_name);
        THROW_INVALID_ARG(error_msg);
    }
}

std::string format_device_mismatch_error(const Device &device1,
                                         const Device &device2,
                                         const std::string &operation_name)
{
    return "Expected all tensors to be on the same device, but found at least two devices, " + device1.to_string() +
           " and " + device2.to_string() + "! (when computing " + operation_name + ")";
}

std::string format_device_type_mismatch_error(const Device &actual_device,
                                              DeviceType expected_device,
                                              const std::string &operation_name)
{
    std::string expected_device_str;
    switch (expected_device)
    {
        case DeviceType::kCPU:
            expected_device_str = "cpu";
            break;
        case DeviceType::kCUDA:
            expected_device_str = "cuda";
            break;
        default:
            expected_device_str = "unknown device type";
            break;
    }

    return "Expected tensor to be on " + expected_device_str + " device, but got " + actual_device.to_string() +
           "! (when computing " + operation_name + ")";
}

}  // namespace validation
}  // namespace origin
