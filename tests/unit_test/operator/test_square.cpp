#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;
/**
 * @brief 平方算子测试类（参数化版本）
 */
class SquareOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(SquareOperatorTest, ForwardBasic)
{
    // 测试基本平方运算
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({1.0f, 4.0f, 9.0f, 16.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor({1.0f, 4.0f, 9.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, ForwardMixedSigns)
{
    // 测试混合符号
    auto x = Tensor({-2.0f, 0.0f, 2.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor({4.0f, 0.0f, 4.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(SquareOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::square(x);
    y.backward();

    // 平方算子的梯度：∂y/∂x = 2x
    auto expected_grad = Tensor({4.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::square(x);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor({4.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, BackwardZeroGradient)
{
    // 测试零点的梯度
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::square(x);
    y.backward();

    auto expected_grad = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, BackwardNegativeValues)
{
    // 测试负值的梯度
    auto x = Tensor({-2.0f, -3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::square(x);
    y.backward();

    // 负值的梯度：2x = 2*(-2) = -4, 2*(-3) = -6
    auto expected_grad = Tensor({-4.0f, -6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(SquareOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 25.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 3.0f);
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    auto expected =
        Tensor(std::vector<float>(100, 9.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

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

TEST_P(SquareOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e5f, 1e-5f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor({1e10f, 1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e-3);
}

TEST_P(SquareOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor({0.01f, 0.04f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(SquareOperatorTest, SmallValues)
{
    // 测试小值
    auto x = Tensor({1e-3f, 2e-6f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor({1e-6f, 4e-12f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e-9);
}

TEST_P(SquareOperatorTest, LargeValues)
{
    // 测试大值
    auto x = Tensor({1e6f, 2e6f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::square(x);

    auto expected = Tensor({1e12f, 4e12f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e9);
}

TEST_P(SquareOperatorTest, IdentityProperty)
{
    // 测试恒等性质：square(square(x)) = x^4
    auto x = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y1 = F::square(x);
    auto y2 = F::square(y1);

    std::vector<float> expected_data;
    expected_data.push_back(2.0f * 2.0f * 2.0f * 2.0f);
    expected_data.push_back(3.0f * 3.0f * 3.0f * 3.0f);
    auto expected = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(y2, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 原地操作测试 ====================

TEST_P(SquareOperatorTest, InplaceBasic)
{
    // 测试基本原地平方运算
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    
    F::square_(x);
    
    auto expected = Tensor({1.0f, 4.0f, 9.0f, 16.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, InplaceZeroTensor)
{
    // 测试零张量原地平方
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x_original = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    
    F::square_(x);
    
    // 结果应该还是零
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x, x_original, origin::test::TestTolerance::kDefault);
}

TEST_P(SquareOperatorTest, InplaceNegativeValues)
{
    // 测试负值原地平方
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    
    F::square_(x);
    
    auto expected = Tensor({1.0f, 4.0f, 9.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x, expected, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(SquareOperatorTest);
