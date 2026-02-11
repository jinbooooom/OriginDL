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
 * @brief 指数算子测试类（参数化版本）
 */
class ExpOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(ExpOperatorTest, ForwardBasic)
{
    // 测试基本指数运算
    auto x = Tensor({0.0f, 1.0f, 2.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {1.0f, static_cast<float>(std::exp(1.0)), static_cast<float>(std::exp(2.0))};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, ForwardZero)
{
    // 测试零值
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    auto expected = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    std::vector<float> expected_data = {static_cast<float>(std::exp(-1.0)), static_cast<float>(std::exp(-2.0))};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, ForwardLargeValues)
{
    // 测试大值
    auto x = Tensor({5.0f, 10.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    std::vector<float> expected_data = {static_cast<float>(std::exp(5.0)), static_cast<float>(std::exp(10.0))};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e3);
}

// ==================== 反向传播测试 ====================

TEST_P(ExpOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::exp(x);
    y.backward();

    // 指数算子的梯度：∂y/∂x = exp(x) = y
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), y, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::exp(x);
    y.backward();

    // 梯度会累积
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), y, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, BackwardZeroGradient)
{
    // 测试零点的梯度
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::exp(x);
    y.backward();

    // exp(0) = 1，所以梯度应该是1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, BackwardNegativeValues)
{
    // 测试负值的梯度
    auto x = Tensor({-1.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::exp(x);
    y.backward();

    // 梯度应该等于y（即exp(x)）
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), y, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(ExpOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), static_cast<float>(std::exp(1.0)), origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 1.0f);
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    float expected_val = static_cast<float>(std::exp(1.0));
    auto expected =
        Tensor(std::vector<float>(100, expected_val), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data;
    for (int i = 0; i < 8; ++i)
    {
        expected_data.push_back(static_cast<float>(std::exp(i)));
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e3);
}

// ==================== 数值稳定性测试 ====================

TEST_P(ExpOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({-10.0f, 0.0f, 10.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    std::vector<float> expected_data = {static_cast<float>(std::exp(-10.0)), 1.0f, static_cast<float>(std::exp(10.0))};
    auto expected                    = Tensor(expected_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e3);
}

TEST_P(ExpOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    std::vector<float> expected_data = {static_cast<float>(std::exp(0.1)), static_cast<float>(std::exp(0.2))};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(ExpOperatorTest, SmallValues)
{
    // 测试小值
    auto x = Tensor({1e-3f, 2e-6f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    std::vector<float> expected_data = {static_cast<float>(std::exp(1e-3)), static_cast<float>(std::exp(2e-6))};
    auto expected                    = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, VerySmallValues)
{
    // 测试非常小的值
    auto x = Tensor({-50.0f, -100.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    // 这些值应该非常接近0
    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_LT(result_data[i], 1e-20f);
    }
}

TEST_P(ExpOperatorTest, IdentityProperty)
{
    // 测试恒等性质：exp(0) = 1
    auto x = Tensor::zeros(Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::exp(x);

    EXPECT_NEAR(result.item<float>(), 1.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, MonotonicProperty)
{
    // 测试单调性：如果 x1 < x2，则 exp(x1) < exp(x2)
    auto x1 = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x2 = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto y1 = F::exp(x1);
    auto y2 = F::exp(x2);

    EXPECT_LT(y1.item<float>(), y2.item<float>());
}

// ==================== 原地操作测试 ====================

TEST_P(ExpOperatorTest, InplaceBasic)
{
    // 测试基本原地指数运算
    auto x = Tensor({0.0f, 1.0f, 2.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    F::exp_(x);

    std::vector<float> expected_data = {1.0f, static_cast<float>(std::exp(1.0)), static_cast<float>(std::exp(2.0))};
    auto expected                    = Tensor(expected_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ExpOperatorTest, InplaceZero)
{
    // 测试零值原地指数运算
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    F::exp_(x);

    auto expected = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x, expected, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(ExpOperatorTest);
