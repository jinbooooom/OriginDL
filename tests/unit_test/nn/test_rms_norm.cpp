#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/nn/layers/rms_norm.h"
#include "origin/operators/normalization/rms_norm.h"
#include "test_utils.h"

using namespace origin;
namespace nn = origin::nn;
namespace F  = origin::functional;

class RMSNormLayerTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(RMSNormLayerTest, BasicForward1D)
{
    // 测试基本的 RMSNorm 层前向传播 - 1D 输入
    nn::RMSNorm rms_norm(4, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 创建输入 (N=2, normalized_shape=4)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto x                    = Tensor(x_data, Shape{2, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = rms_norm.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 4U);

    // 验证输出不为 NaN 或 Inf
    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormLayerTest, BasicForward2D)
{
    // 测试 RMSNorm 层前向传播 - 2D 输入 (N, H, normalized_shape)
    nn::RMSNorm rms_norm(3, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 创建输入 (N=2, H=4, normalized_shape=3)
    std::vector<float> x_data(2 * 4 * 3, 1.0f);
    auto x = Tensor(x_data, Shape{2, 4, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = rms_norm.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 3U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 4U);
    EXPECT_EQ(y.shape()[2], 3U);

    // 验证输出不为 NaN 或 Inf
    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormLayerTest, BasicForward3D)
{
    // 测试 RMSNorm 层前向传播 - 3D 输入 (N, H, W, normalized_shape)
    nn::RMSNorm rms_norm(2, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 创建输入 (N=2, H=3, W=4, normalized_shape=2)
    std::vector<float> x_data(2 * 3 * 4 * 2, 1.0f);
    auto x = Tensor(x_data, Shape{2, 3, 4, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = rms_norm.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 3U);
    EXPECT_EQ(y.shape()[2], 4U);
    EXPECT_EQ(y.shape()[3], 2U);

    // 验证输出不为 NaN 或 Inf
    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(RMSNormLayerTest, SingleElement)
{
    // 测试单个元素的情况
    nn::RMSNorm rms_norm(1, 1e-5f);
    rms_norm.to(Device(deviceType()));

    std::vector<float> x_data = {2.0f};
    auto x                    = Tensor(x_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({1}));

    // 对于单个元素，RMSNorm 应该输出 gamma * x / sqrt(x^2 + eps)
    // gamma 初始化为 1，所以输出应该接近 1 (因为 x / sqrt(x^2 + eps) ≈ 1)
    auto y_data = y.to_vector<float>();
    EXPECT_NEAR(y_data[0], 1.0f, 0.01f);
}

TEST_P(RMSNormLayerTest, ResetParameters)
{
    // 测试参数重置
    nn::RMSNorm rms_norm(4, 1e-5f);
    rms_norm.to(Device(deviceType()));

    // 修改 gamma 参数
    auto weight      = rms_norm.weight();
    auto weight_data = weight->data_ptr<float>();
    weight_data[0]   = 2.0f;

    // 重置参数
    rms_norm.reset_parameters();

    // 验证 gamma 被重置为全 1
    auto reset_weight = rms_norm.weight();
    auto reset_data   = reset_weight->to_vector<float>();
    for (float val : reset_data)
    {
        EXPECT_FLOAT_EQ(val, 1.0f);
    }
}

INSTANTIATE_TEST_SUITE_P(RMSNormLayerTests, RMSNormLayerTest, ::testing::Values(DeviceType::kCPU));

// ==================== CUDA 特定测试 ====================

#ifdef WITH_CUDA
class RMSNormCUDATest : public ::testing::Test
{};

TEST_F(RMSNormCUDATest, ValidNormalizedShapesForward)
{
    // 测试 CUDA 支持的有效 normalized_shape 值 - 前向传播
    std::vector<size_t> valid_shapes = {64, 128, 256, 512, 1024};

    for (size_t normalized_shape : valid_shapes)
    {
        nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
        rms_norm.to(Device(DeviceType::kCUDA));

        // 创建输入 (N=2, normalized_shape)
        std::vector<float> x_data(2 * normalized_shape, 1.0f);
        auto x = Tensor(x_data, Shape{2, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

        // 前向传播 - 不应该抛出异常
        auto y = rms_norm.forward(x);

        EXPECT_EQ(y.shape(), Shape({2, normalized_shape}));

        // 验证输出不为 NaN 或 Inf
        auto y_data = y.to_vector<float>();
        for (float val : y_data)
        {
            EXPECT_FALSE(std::isnan(val));
            EXPECT_FALSE(std::isinf(val));
        }
    }
}

TEST_F(RMSNormCUDATest, Forward64)
{
    // 测试 normalized_shape=64 的前向传播
    size_t normalized_shape = 64;
    size_t batch_size       = 4;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape);
    for (size_t i = 0; i < batch_size * normalized_shape; ++i)
    {
        x_data[i] = static_cast<float>((i % 10) + 1);
    }

    auto x = Tensor(x_data, Shape{batch_size, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({batch_size, normalized_shape}));

    auto y_data = y.to_vector<float>();
    for (float val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(RMSNormCUDATest, Forward128)
{
    // 测试 normalized_shape=128 的前向传播
    size_t normalized_shape = 128;
    size_t batch_size       = 8;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, Forward256)
{
    // 测试 normalized_shape=256 的前向传播
    size_t normalized_shape = 256;
    size_t batch_size       = 16;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, Forward512)
{
    // 测试 normalized_shape=512 的前向传播
    size_t normalized_shape = 512;
    size_t batch_size       = 4;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, Forward1024)
{
    // 测试 normalized_shape=1024 的前向传播
    size_t normalized_shape = 1024;
    size_t batch_size       = 2;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, ForwardNumericalCorrectness)
{
    // 测试 CUDA 前向传播数值正确性
    size_t normalized_shape = 256;
    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    // 创建输入 (N=4, normalized_shape=256)
    std::vector<float> x_data;
    for (size_t i = 0; i < 4 * normalized_shape; ++i)
    {
        x_data.push_back(static_cast<float>((i % 10) + 1));
    }
    auto x = Tensor(x_data, Shape{4, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    // 手动计算第一个 token 的 RMS
    float sum_sq = 0.0f;
    for (size_t i = 0; i < normalized_shape; ++i)
    {
        float val = x_data[i];
        sum_sq += val * val;
    }
    float rms_first_token = std::sqrt(sum_sq / normalized_shape + 1e-5f);

    auto y_data = y.to_vector<float>();
    EXPECT_NEAR(y_data[0], x_data[0] / rms_first_token, 1e-3f);
}

TEST_F(RMSNormCUDATest, Backward64)
{
    // 测试 normalized_shape=64 的反向传播
    size_t normalized_shape = 64;
    size_t batch_size       = 4;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape);
    for (size_t i = 0; i < batch_size * normalized_shape; ++i)
    {
        x_data[i] = static_cast<float>((i % 10) + 1);
    }

    auto x = Tensor(x_data, Shape{batch_size, normalized_shape},
                    dtype(DataType::kFloat32).device(DeviceType::kCUDA).requires_grad(true));

    auto y = rms_norm.forward(x);

    // 计算损失
    auto loss = F::sum(y);

    // 反向传播
    loss.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), Shape({batch_size, normalized_shape}));

    auto gx_data = x.grad().to_vector<float>();
    for (float val : gx_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(RMSNormCUDATest, Backward128)
{
    // 测试 normalized_shape=128 的反向传播
    size_t normalized_shape = 128;
    size_t batch_size       = 8;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape},
                    dtype(DataType::kFloat32).device(DeviceType::kCUDA).requires_grad(true));

    auto y = rms_norm.forward(x);
    auto loss = F::sum(y);
    loss.backward();

    EXPECT_EQ(x.grad().shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, Backward256)
{
    // 测试 normalized_shape=256 的反向传播
    size_t normalized_shape = 256;
    size_t batch_size       = 16;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape},
                    dtype(DataType::kFloat32).device(DeviceType::kCUDA).requires_grad(true));

    auto y = rms_norm.forward(x);
    auto loss = F::sum(y);
    loss.backward();

    EXPECT_EQ(x.grad().shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, Backward512)
{
    // 测试 normalized_shape=512 的反向传播
    size_t normalized_shape = 512;
    size_t batch_size       = 4;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape},
                    dtype(DataType::kFloat32).device(DeviceType::kCUDA).requires_grad(true));

    auto y = rms_norm.forward(x);
    auto loss = F::sum(y);
    loss.backward();

    EXPECT_EQ(x.grad().shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, Backward1024)
{
    // 测试 normalized_shape=1024 的反向传播
    size_t normalized_shape = 1024;
    size_t batch_size       = 2;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape},
                    dtype(DataType::kFloat32).device(DeviceType::kCUDA).requires_grad(true));

    auto y = rms_norm.forward(x);
    auto loss = F::sum(y);
    loss.backward();

    EXPECT_EQ(x.grad().shape(), Shape({batch_size, normalized_shape}));
}

TEST_F(RMSNormCUDATest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    size_t normalized_shape = 256;
    size_t batch_size       = 4;

    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape},
                    dtype(DataType::kFloat32).device(DeviceType::kCUDA).requires_grad(true));

    auto y = rms_norm.forward(x);

    // 计算损失并反向传播
    auto loss = F::sum(F::square(y - Tensor::ones(y.shape(), dtype(DataType::kFloat32).device(DeviceType::kCUDA))));
    loss.backward();

    // 验证梯度
    EXPECT_EQ(x.grad().shape(), Shape({batch_size, normalized_shape}));

    auto gx_data = x.grad().to_vector<float>();
    for (float val : gx_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_F(RMSNormCUDATest, LargeInput)
{
    // 测试大批量输入
    size_t normalized_shape = 512;
    size_t batch_size       = 128;
    nn::RMSNorm rms_norm(normalized_shape, 1e-5f);
    rms_norm.to(Device(DeviceType::kCUDA));

    std::vector<float> x_data(batch_size * normalized_shape, 1.0f);
    auto x = Tensor(x_data, Shape{batch_size, normalized_shape}, dtype(DataType::kFloat32).device(DeviceType::kCUDA));

    auto y = rms_norm.forward(x);

    EXPECT_EQ(y.shape(), Shape({batch_size, normalized_shape}));

    auto y_data    = y.to_vector<float>();
    float expected = 1.0f;
    for (float val : y_data)
    {
        EXPECT_NEAR(val, expected, 1e-4f);
    }
}

TEST_F(RMSNormCUDATest, Float64Support)
{
    // 测试 float64 支持
    size_t normalized_shape = 256;

    std::vector<double> x_data(2 * normalized_shape, 1.0);
    auto x = Tensor(x_data, Shape{2, normalized_shape}, dtype(DataType::kFloat64).device(DeviceType::kCUDA));

    std::vector<double> gamma_data(normalized_shape, 1.0);
    auto gamma = Tensor(gamma_data, Shape{normalized_shape}, dtype(DataType::kFloat64).device(DeviceType::kCUDA));

    auto y = F::rms_norm(x, gamma, 1e-5);

    EXPECT_EQ(y.shape(), Shape({2, normalized_shape}));

    auto y_data = y.to_vector<double>();
    for (double val : y_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

#endif  // WITH_CUDA
