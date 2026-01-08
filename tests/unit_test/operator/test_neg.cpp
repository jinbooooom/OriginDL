#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
/**
 * @brief 取反算子测试类（参数化版本）
 */
class NegOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(NegOperatorTest, ForwardBasic)
{
    // 测试基本负号运算
    auto x = Tensor({1.0f, -2.0f, 3.0f, -4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x = Tensor({2.0f, -3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = -x;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({-2.0f, 3.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    // 结果应该等于x（零的负号还是零）
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, ForwardPositiveValues)
{
    // 测试正值
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    auto expected = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    auto expected = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(NegOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = neg(x);
    y.backward();

    // 负号算子的梯度：∂y/∂x = -1
    auto expected_grad = Tensor({-1.0f, -1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = neg(x);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor({-1.0f, -1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = neg(x);
    y.backward();

    auto expected_grad =
        Tensor({-1.0f, -1.0f, -1.0f, -1.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(NegOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), -5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 2.0f);
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    auto expected =
        Tensor(std::vector<float>(100, -2.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data;
    for (int i = 1; i <= 8; ++i)
    {
        expected_data.push_back(-static_cast<float>(i));
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(NegOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10f, 1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    auto expected = Tensor({-1e10f, -1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    auto expected = Tensor({-0.1f, -0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(NegOperatorTest, DoubleNegation)
{
    // 测试双重负号
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(neg(x));

    // 双重负号应该等于原值
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(NegOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f, 0.0f, -4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = neg(x);

    auto expected = Tensor({-1.0f, 2.0f, 0.0f, 4.0f, -5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(NegOperatorTest);
