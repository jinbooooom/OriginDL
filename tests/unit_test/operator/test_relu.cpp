#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"
#include "origin.h"

using namespace origin;

/**
 * @brief ReLU 算子测试类（参数化版本）
 */
class ReLUOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(ReLUOperatorTest, ForwardBasic)
{
    // 测试基本 ReLU 运算
    // 输入: [-1.0, 0.0, 1.0, 2.0]
    // 预期: [0.0, 0.0, 1.0, 2.0]
    auto x = Tensor({-1.0f, 0.0f, 1.0f, 2.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    Shape expected_shape{4};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {0.0f, 0.0f, 1.0f, 2.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, ForwardAllPositive)
{
    // 测试全正数
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    // ReLU 应该保持不变
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, ForwardAllNegative)
{
    // 测试全负数
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    // ReLU 应该全部变为 0
    auto expected = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, ForwardZero)
{
    // 测试零值
    auto x = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    // ReLU(0) = 0
    auto expected = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量
    auto x = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {0.0f, 2.0f, 0.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(ReLUOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({-1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = relu(x);
    y.backward();

    // 梯度：x > 0 时为 1，x <= 0 时为 0
    std::vector<float> expected_grad_data = {0.0f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    // 注意：y.backward() 会自动使用全1的梯度，所以这里我们验证基本行为
    auto x = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{4},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = relu(x);
    y.backward();

    // 梯度：gx = gy * (x > 0 ? 1 : 0)
    // 当 gy = 1（默认）时，gx = (x > 0 ? 1 : 0)
    // x = [-1, 2, -3, 4]
    // gx = [0, 1, 0, 1]
    std::vector<float> expected_grad_data = {0.0f, 1.0f, 0.0f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, BackwardZero)
{
    // 测试 x=0 的梯度
    auto x = Tensor({0.0f, 0.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = relu(x);
    y.backward();

    // ReLU 在 x=0 处的梯度为 0（通常定义）
    auto expected_grad = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, BackwardTwoDimensional)
{
    // 测试 2D 张量的反向传播
    auto x = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = relu(x);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度值
    std::vector<float> expected_grad_data = {0.0f, 1.0f, 0.0f, 1.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(ReLUOperatorTest, SingleElement)
{
    // 测试单个元素
    auto x = Tensor({-5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 0.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, SingleElementPositive)
{
    // 测试单个正元素
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100);
    for (size_t i = 0; i < 100; ++i)
    {
        data[i] = static_cast<float>(i) - 50.0f;  // 从 -50 到 49
    }
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证前 50 个元素为 0（对应 -50 到 -1），索引 50 为 0（对应 0），后 49 个元素为正数（对应 1 到 49）
    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i <= 50; ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0f, origin::test::TestTolerance::kDefault);
    }
    for (size_t i = 51; i < 100; ++i)
    {
        EXPECT_GT(result_data[i], 0.0f);
        // 验证值正确：ReLU(i - 50) = i - 50（当 i > 50 时）
        EXPECT_NEAR(result_data[i], static_cast<float>(i - 50), origin::test::TestTolerance::kDefault);
    }
}

TEST_P(ReLUOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    std::vector<float> input_data = {-1.0f, 2.0f, -3.0f, 4.0f, -5.0f, 6.0f, -7.0f, 8.0f};
    auto x                        = Tensor(input_data, Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {0.0f, 2.0f, 0.0f, 4.0f, 0.0f, 6.0f, 0.0f, 8.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(ReLUOperatorTest, IdentityProperty)
{
    // 测试恒等性质：对于正数，ReLU(x) = x
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(ReLUOperatorTest, ZeroProperty)
{
    // 测试零性质：对于负数，ReLU(x) = 0
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = relu(x);

    auto expected = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(ReLUOperatorTest);
