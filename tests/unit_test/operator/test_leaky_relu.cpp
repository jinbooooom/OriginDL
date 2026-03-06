#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief LeakyReLU 算子测试类（参数化版本）
 */
class LeakyReLUOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(LeakyReLUOperatorTest, ForwardBasic)
{
    // 测试基本 LeakyReLU 运算 (alpha=0.1)
    // 输入: [-1.0, -2.0, 1.0, 2.0]
    // 预期: [-0.1, -0.2, 1.0, 2.0]
    auto x = Tensor({-1.0f, -2.0f, 1.0f, 2.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    Shape expected_shape{4};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {-0.1f, -0.2f, 1.0f, 2.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, ForwardAllPositive)
{
    // 测试全正数（正数部分应该保持不变）
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    // LeakyReLU 对于正数应该保持不变
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, ForwardAllNegative)
{
    // 测试全负数（负数部分应该乘以 alpha）
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    // LeakyReLU 对于负数应该乘以 alpha
    std::vector<float> expected_data = {-0.1f, -0.2f, -0.3f};
    auto expected                    = Tensor(expected_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, ForwardZero)
{
    // 测试零值
    auto x = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    // LeakyReLU(0) = 0（0 既不是正数也不是负数，按公式应该乘以 alpha，但 0 * alpha = 0）
    auto expected = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量
    auto x = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {-0.1f, 2.0f, -0.3f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, ForwardDifferentAlpha)
{
    // 测试不同的 alpha 值
    auto x = Tensor({-1.0f, -2.0f, 1.0f, 2.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));

    // alpha = 0.01
    auto result1                      = F::leaky_relu(x, 0.01f);
    std::vector<float> expected_data1 = {-0.01f, -0.02f, 1.0f, 2.0f};
    auto expected1 = Tensor(expected_data1, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, expected1, origin::test::TestTolerance::kDefault);

    // alpha = 0.5
    auto result2                      = F::leaky_relu(x, 0.5f);
    std::vector<float> expected_data2 = {-0.5f, -1.0f, 1.0f, 2.0f};
    auto expected2 = Tensor(expected_data2, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result2, expected2, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, SingleElement)
{
    // 测试单个元素
    auto x = Tensor({-5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), -0.5f, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, SingleElementPositive)
{
    // 测试单个正元素
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100);
    for (size_t i = 0; i < 100; ++i)
    {
        data[i] = static_cast<float>(i) - 50.0f;  // 从 -50 到 49
    }
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证结果
    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < 100; ++i)
    {
        float expected_val = (data[i] > 0) ? data[i] : data[i] * 0.1f;
        EXPECT_NEAR(result_data[i], expected_val, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(LeakyReLUOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
    auto x                        = Tensor(input_data, Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {-0.1f, 2.0f, -0.3f, 4.0f, -0.5f, 6.0f, -0.7f, 8.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, IdentityProperty)
{
    // 测试恒等性质：对于正数，LeakyReLU(x) = x
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, NegativeProperty)
{
    // 测试负数性质：对于负数，LeakyReLU(x) = alpha * x
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.2f);

    // LeakyReLU 对于负数应该等于 alpha * x
    std::vector<float> expected_data = {-0.2f, -0.4f, -0.6f};
    auto expected                    = Tensor(expected_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(LeakyReLUOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    // LeakyReLU 的梯度：当 x > 0 时为 1，当 x <= 0 时为 alpha
    auto x = Tensor({-1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::leaky_relu(x, 0.1f);
    y.backward();

    // 梯度：x > 0 时为 1，x <= 0 时为 0.1
    std::vector<float> expected_grad_data = {0.1f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{4},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::leaky_relu(x, 0.1f);
    y.backward();

    // 梯度：gx = gy * (x > 0 ? 1 : alpha)
    // 当 gy = 1（默认）时，gx = (x > 0 ? 1 : 0.1)
    // x = [-1, 2, -3, 4]
    // gx = [0.1, 1, 0.1, 1]
    std::vector<float> expected_grad_data = {0.1f, 1.0f, 0.1f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, BackwardZero)
{
    // 测试 x=0 的梯度
    auto x = Tensor({0.0f, 0.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::leaky_relu(x, 0.1f);
    y.backward();

    // LeakyReLU 在 x=0 处的梯度为 alpha（通常定义）
    std::vector<float> expected_grad_data = {0.1f, 0.1f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, BackwardTwoDimensional)
{
    // 测试 2D 张量的反向传播
    auto x = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::leaky_relu(x, 0.1f);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度值
    std::vector<float> expected_grad_data = {0.1f, 1.0f, 0.1f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, BackwardDifferentAlpha)
{
    // 测试不同 alpha 值的梯度
    auto x = Tensor({-1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    // alpha = 0.2
    auto y = F::leaky_relu(x, 0.2f);
    y.backward();

    // 梯度：x > 0 时为 1，x <= 0 时为 0.2
    std::vector<float> expected_grad_data = {0.2f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, BackwardChainRule)
{
    // 测试链式法则
    auto x = Tensor({-1.0f, 2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::leaky_relu(x, 0.1f);
    // 对输出再进行一次操作
    auto z = y * 2.0f;
    z.backward();

    // 梯度：dz/dx = dz/dy * dy/dx = 2 * (x > 0 ? 1 : 0.1)
    // x = [-1, 2, -3]
    // dy/dx = [0.1, 1, 0.1]
    // dz/dx = [0.2, 2, 0.2]
    std::vector<float> expected_grad_data = {0.2f, 2.0f, 0.2f};
    auto expected_grad = Tensor(expected_grad_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(LeakyReLUOperatorTest, LargeAlpha)
{
    // 测试较大的 alpha 值
    auto x = Tensor({-1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.5f);

    std::vector<float> expected_data = {-0.5f, 2.0f};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LeakyReLUOperatorTest, SmallAlpha)
{
    // 测试较小的 alpha 值
    auto x = Tensor({-1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.001f);

    std::vector<float> expected_data = {-0.001f, 2.0f};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kStrict);
}

TEST_P(LeakyReLUOperatorTest, MixedValues)
{
    // 测试混合正负值
    auto x = Tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    std::vector<float> expected_data = {-0.2f, -0.1f, 0.0f, 1.0f, 2.0f};
    auto expected                    = Tensor(expected_data, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(LeakyReLUOperatorTest, VerySmallValues)
{
    // 测试非常小的值
    auto x = Tensor({-1e-10f, 1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    // -1e-10 * 0.1 = -1e-11, 1e-10 保持不变
    std::vector<float> expected_data = {-1e-11f, 1e-10f};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, 1e-12);  // 使用更小的容差
}

TEST_P(LeakyReLUOperatorTest, VeryLargeValues)
{
    // 测试非常大的值
    auto x = Tensor({-1000.0f, 1000.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.1f);

    std::vector<float> expected_data = {-100.0f, 1000.0f};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(LeakyReLUOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({-100.0f, -10.0f, -1.0f, 0.0f, 1.0f, 10.0f, 100.0f}, Shape{7},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::leaky_relu(x, 0.01f);

    std::vector<float> expected_data = {-1.0f, -0.1f, -0.01f, 0.0f, 1.0f, 10.0f, 100.0f};
    auto expected                    = Tensor(expected_data, Shape{7}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(LeakyReLUOperatorTest);
