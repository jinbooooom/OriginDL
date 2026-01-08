#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
/**
 * @brief 求和算子测试类（参数化版本）
 */
class SumOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(SumOperatorTest, ForwardBasic)
{
    // 测试基本求和运算
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);  // 标量结果
    EXPECT_NEAR(result.item<float>(), 10.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardOneDimensional)
{
    // 测试一维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 15.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 0.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0f, -2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 4.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardSingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardWithAxis)
{
    // 测试指定轴的求和
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 沿轴0求和（列求和）
    auto result0 = sum(x, 0);
    EXPECT_EQ(result0.shape(), Shape{3});
    auto expected0 = Tensor({5.0f, 7.0f, 9.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result0, expected0, origin::test::TestTolerance::kDefault);

    // 沿轴1求和（行求和）
    auto result1 = sum(x, 1);
    EXPECT_EQ(result1.shape(), Shape{2});
    auto expected1 = Tensor({6.0f, 15.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, expected1, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(SumOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sum(x);
    y.backward();

    // 求和算子的梯度：∂y/∂x = 1（广播到所有元素）
    auto expected_grad = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sum(x);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, BackwardWithAxis)
{
    // 测试带轴的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sum(x, 0);  // 沿轴0求和
    y.backward();

    // 梯度应该广播回原始形状
    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sum(x);
    y.backward();

    auto expected_grad = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(SumOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(1000, 1.0f);
    auto x = Tensor(data, Shape{100, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 1000.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 36.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ThreeDimensionalWithAxis)
{
    // 测试三维张量带轴求和
    auto x = Tensor({1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                     13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f},
                    Shape{4, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 沿轴0求和
    auto result0 = sum(x, 0);
    Shape expected_shape0{3, 2};
    EXPECT_EQ(result0.shape(), expected_shape0);
    auto expected0 = Tensor({40.0f, 44.0f, 48.0f, 52.0f, 56.0f, 60.0f}, expected_shape0,
                            dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result0, expected0, origin::test::TestTolerance::kDefault);

    // 沿轴1求和
    auto result1 = sum(x, 1);
    Shape expected_shape1{4, 2};
    EXPECT_EQ(result1.shape(), expected_shape1);
    auto expected1 = Tensor({9.0f, 12.0f, 27.0f, 30.0f, 45.0f, 48.0f, 63.0f, 66.0f}, expected_shape1,
                            dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, expected1, origin::test::TestTolerance::kDefault);

    // 沿轴2求和
    auto result2 = sum(x, 2);
    Shape expected_shape2{4, 3};
    EXPECT_EQ(result2.shape(), expected_shape2);
    auto expected2 = Tensor({3.0f, 7.0f, 11.0f, 15.0f, 19.0f, 23.0f, 27.0f, 31.0f, 35.0f, 39.0f, 43.0f, 47.0f},
                            expected_shape2, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result2, expected2, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardOneByOneSumAxis0)
{
    // 测试 (1,1) 形状的张量，对轴0求和
    // 这个测试用于验证conv2d_backward中gb计算的问题
    auto x = Tensor({5.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 沿轴0求和，应该得到 (1,) 形状
    auto result = sum(x, 0);
    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardOneByOneSumAxis1)
{
    // 测试 (1,1) 形状的张量，对轴1求和
    auto x = Tensor({5.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 沿轴1求和，应该得到 (1,) 形状
    auto result = sum(x, 1);
    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, ForwardMultipleSumsOnOneByOne)
{
    // 测试对 (1,1) 形状连续求和，模拟conv2d_backward中gb的计算过程
    // gy形状: (1, 1, 1, 1) -> sum(2) -> (1, 1, 1) -> sum(2) -> (1, 1) -> sum(0) -> (1,)
    auto gy = Tensor({1.0f}, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 第一步：sum(2) on (1,1,1,1) -> (1,1,1)
    auto step1 = sum(gy, 2);
    Shape expected_shape1{1, 1, 1};
    EXPECT_EQ(step1.shape(), expected_shape1);

    // 第二步：sum(2) on (1,1,1) -> (1,1)
    auto step2 = sum(step1, 2);
    Shape expected_shape2{1, 1};
    EXPECT_EQ(step2.shape(), expected_shape2);

    // 第三步：sum(0) on (1,1) -> (1,)
    auto step3 = sum(step2, 0);
    Shape expected_shape3{1};
    EXPECT_EQ(step3.shape(), expected_shape3);
    EXPECT_NEAR(step3.item<float>(), 1.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(SumOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10f, 1e-10f, -1e10f, -1e-10f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    EXPECT_NEAR(result.item<float>(), 0.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f, 0.3f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    EXPECT_NEAR(result.item<float>(), 0.6f, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(SumOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f, 3.0f, -4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    EXPECT_NEAR(result.item<float>(), 3.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, IdentityProperty)
{
    // 测试恒等性质：sum(x) = x（当x是标量时）
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum(x);

    EXPECT_NEAR(result.item<float>(), x.item<float>(), origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, CommutativeProperty)
{
    // 测试交换性质：sum(a + b) = sum(a) + sum(b)
    auto a = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto sum_ab           = sum(a + b);
    auto sum_a_plus_sum_b = sum(a) + sum(b);

    EXPECT_NEAR(sum_ab.item<float>(), sum_a_plus_sum_b.item<float>(), origin::test::TestTolerance::kDefault);
}

TEST_P(SumOperatorTest, AssociativeProperty)
{
    // 测试结合性质：sum(a + b + c) = sum(a) + sum(b) + sum(c)
    auto a = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto c = Tensor({5.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto sum_abc                     = sum(a + b + c);
    auto sum_a_plus_sum_b_plus_sum_c = sum(a) + sum(b) + sum(c);

    EXPECT_NEAR(sum_abc.item<float>(), sum_a_plus_sum_b_plus_sum_c.item<float>(),
                origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(SumOperatorTest);
