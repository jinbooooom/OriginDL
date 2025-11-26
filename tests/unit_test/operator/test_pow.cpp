#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;
/**
 * @brief 幂算子测试类（参数化版本）
 */
class PowOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(PowOperatorTest, ForwardBasic)
{
    // 测试基本幂运算
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = pow(x, exponent);

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({4.0f, 9.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = x ^ exponent;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({8.0f, 27.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardZeroExponent)
{
    // 测试零指数
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 0;

    auto result = pow(x, exponent);

    auto expected = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardNegativeExponent)
{
    // 测试负指数
    auto x       = Tensor({2.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = -1;

    auto result = pow(x, exponent);

    auto expected = Tensor({0.5f, 0.25f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardZeroBase)
{
    // 测试零底数
    auto x       = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = pow(x, exponent);

    auto expected = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(PowOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = 2;

    auto y = pow(x, exponent);
    y.backward();

    // 幂算子的梯度：∂y/∂x = exponent * x^(exponent-1)
    auto expected_grad = Tensor({4.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = 3;

    auto y = pow(x, exponent);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor({12.0f, 27.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, BackwardZeroExponent)
{
    // 测试零指数的梯度
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = 0;

    auto y = pow(x, exponent);
    y.backward();

    auto expected_grad = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, BackwardNegativeExponent)
{
    // 测试负指数的梯度
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = -2;

    auto y = pow(x, exponent);
    y.backward();

    // 梯度：-2 * x^(-3) = -2 / x^3
    auto expected_grad = Tensor({-2.0f / 8.0f, -2.0f / 27.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(PowOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x       = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = pow(x, exponent);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 8.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 2.0f);
    auto x       = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = pow(x, exponent);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    
    auto expected = Tensor(std::vector<float>(100, 4.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = pow(x, exponent);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data;
    for (int i = 1; i <= 8; ++i)
    {
        float val = static_cast<float>(i);
        expected_data.push_back(val * val);
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(PowOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x       = Tensor({1e-3f, 1e5f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = pow(x, exponent);

    auto expected = Tensor({1e-6f, 1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e9);
}

TEST_P(PowOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x       = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = pow(x, exponent);

    auto expected = Tensor({0.001f, 0.008f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(PowOperatorTest, DifferentExponents)
{
    // 测试不同指数
    auto x = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    // 测试指数为1
    auto result1 = pow(x, 1);
    EXPECT_NEAR(result1.item<float>(), 2.0f, origin::test::TestTolerance::kDefault);

    // 测试指数为2
    auto result2 = pow(x, 2);
    EXPECT_NEAR(result2.item<float>(), 4.0f, origin::test::TestTolerance::kDefault);

    // 测试指数为3
    auto result3 = pow(x, 3);
    EXPECT_NEAR(result3.item<float>(), 8.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, IdentityProperty)
{
    // 测试恒等性质：x^1 = x
    auto x = Tensor({2.0f, 3.0f, 4.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = pow(x, 1);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ZeroPowerProperty)
{
    // 测试零幂性质：x^0 = 1
    auto x = Tensor({2.0f, 3.0f, 4.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = pow(x, 0);

    auto expected = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, LargeExponent)
{
    // 测试大指数
    auto x       = Tensor({1.1f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 10;

    auto result = pow(x, exponent);

    EXPECT_NEAR(result.item<float>(), static_cast<float>(std::pow(1.1, 10)), origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(PowOperatorTest);
