#include "gtest_utils.h"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

namespace origin
{
namespace test
{

void GTestUtils::EXPECT_TENSORS_EQ(const Tensor &a, const Tensor &b, double tolerance)
{
    // 检查形状
    if (a.shape() != b.shape())
    {
        std::ostringstream oss;
        oss << "Tensor shapes do not match: " << a.shape().to_string() << " vs " << b.shape().to_string();
        ADD_FAILURE() << oss.str();
        return;
    }

    // 检查数据：统一转换到 float32 再比较，避免不同dtype导致的问题
    Tensor a_f = (a.dtype() == DataType::kFloat32) ? a : a.to(DataType::kFloat32);
    Tensor b_f = (b.dtype() == DataType::kFloat32) ? b : b.to(DataType::kFloat32);

    auto data_a = a_f.to_vector<float>();
    auto data_b = b_f.to_vector<float>();

    if (data_a.size() != data_b.size())
    {
        ADD_FAILURE() << "Tensor data sizes do not match: " << data_a.size() << " vs " << data_b.size();
        return;
    }

    // 逐元素比较
    bool all_equal        = true;
    size_t first_mismatch = 0;
    for (size_t i = 0; i < data_a.size(); ++i)
    {
        if (!TestUtils::isEqual(data_a[i], data_b[i], tolerance))
        {
            if (all_equal)
            {
                first_mismatch = i;
            }
            all_equal = false;
        }
    }

    if (!all_equal)
    {
        std::ostringstream oss;
        oss << "Tensors are not equal within tolerance " << tolerance << ". "
            << "First mismatch at index " << first_mismatch << ": " << data_a[first_mismatch] << " vs "
            << data_b[first_mismatch];
        ADD_FAILURE() << oss.str();
    }
}

void GTestUtils::EXPECT_TENSORS_NEAR(const Tensor &a, const Tensor &b, double rtol, double atol)
{
    // 检查形状
    if (a.shape() != b.shape())
    {
        std::ostringstream oss;
        oss << "Tensor shapes do not match: " << a.shape().to_string() << " vs " << b.shape().to_string();
        ADD_FAILURE() << oss.str();
        return;
    }

    // 检查数据：统一转换到 float32 再比较
    Tensor a_f = (a.dtype() == DataType::kFloat32) ? a : a.to(DataType::kFloat32);
    Tensor b_f = (b.dtype() == DataType::kFloat32) ? b : b.to(DataType::kFloat32);

    auto data_a = a_f.to_vector<float>();
    auto data_b = b_f.to_vector<float>();

    if (data_a.size() != data_b.size())
    {
        ADD_FAILURE() << "Tensor data sizes do not match: " << data_a.size() << " vs " << data_b.size();
        return;
    }

    // 逐元素比较
    bool all_near         = true;
    size_t first_mismatch = 0;
    double max_diff       = 0.0;
    for (size_t i = 0; i < data_a.size(); ++i)
    {
        double diff      = std::abs(data_a[i] - data_b[i]);
        double max_val   = std::max(std::abs(data_a[i]), std::abs(data_b[i]));
        double threshold = atol + rtol * max_val;

        if (diff > threshold)
        {
            if (all_near)
            {
                first_mismatch = i;
                max_diff       = diff;
            }
            all_near = false;
        }
    }

    if (!all_near)
    {
        std::ostringstream oss;
        oss << "Tensors are not near within tolerance (rtol=" << rtol << ", atol=" << atol << "). "
            << "First mismatch at index " << first_mismatch << ": " << data_a[first_mismatch] << " vs "
            << data_b[first_mismatch] << " (diff=" << max_diff << ")";
        ADD_FAILURE() << oss.str();
    }
}

void GTestUtils::EXPECT_TENSOR_SHAPE(const Tensor &tensor, const Shape &expected_shape)
{
    if (tensor.shape() != expected_shape)
    {
        std::ostringstream oss;
        oss << "Tensor shape mismatch: expected " << expected_shape.to_string() << ", got "
            << tensor.shape().to_string();
        ADD_FAILURE() << oss.str();
    }
}

void GTestUtils::EXPECT_TENSOR_DEVICE(const Tensor &tensor, DeviceType expected_device)
{
    if (tensor.device().type() != expected_device)
    {
        std::ostringstream oss;
        std::string expected_str = (expected_device == DeviceType::kCPU) ? "CPU" : "CUDA";
        oss << "Tensor device mismatch: expected " << expected_str << ", got " << tensor.device().to_string();
        ADD_FAILURE() << oss.str();
    }
}

void GTestUtils::EXPECT_TENSOR_DEVICE(const Tensor &tensor, const Device &expected_device)
{
    if (tensor.device() != expected_device)
    {
        std::ostringstream oss;
        oss << "Tensor device mismatch: expected " << expected_device.to_string() << ", got "
            << tensor.device().to_string();
        ADD_FAILURE() << oss.str();
    }
}

}  // namespace test
}  // namespace origin
