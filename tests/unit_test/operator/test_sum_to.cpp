#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
/**
 * @brief sum_to算子测试类（参数化版本）
 */
class SumToOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(SumToOperatorTest, ForwardBasic)
{
    // 测试基本sum_to运算
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_NEAR(result.item<float>(), 10.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardToScalar)
{
    // 测试求和到标量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 15.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardToSameShape)
{
    // 测试相同形状（应该不变）
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = sum_to(x, target_shape);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardToLargerShape)
{
    // 测试到更大形状（应该抛出异常，因为libtorch不支持广播）
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3};

    EXPECT_THROW(sum_to(x, target_shape), std::runtime_error);
}

TEST_P(SumToOperatorTest, ForwardToSmallerShape)
{
    // 测试到更小形状（应该求和）
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 1};

    auto result = sum_to(x, target_shape);

    auto expected = Tensor({6.0f, 15.0f}, Shape{2, 1}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 0.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(SumToOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto y = sum_to(x, target_shape);
    y.backward();

    // sum_to算子的梯度：∂y/∂x = 1（广播回原始形状）
    auto expected_grad = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto y = sum_to(x, target_shape);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, BackwardToSameShape)
{
    // 测试相同形状的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto y = sum_to(x, target_shape);
    y.backward();

    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, BackwardToLargerShape)
{
    // 测试到更大形状的反向传播（应该抛出异常）
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3};

    EXPECT_THROW(sum_to(x, target_shape), std::runtime_error);
}

// ==================== 边界情况测试 ====================

TEST_P(SumToOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 1.0f);
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 100.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 36.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ThreeDimensionalTo2D)
{
    // 测试三维张量到二维
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({10.0f, 26.0f}, Shape{2, 1}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(SumToOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10f, 1e-10f, -1e10f, -1e-10f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_NEAR(result.item<float>(), 0.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f, 0.3f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = sum_to(x, target_shape);

    EXPECT_NEAR(result.item<float>(), 0.6f, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(SumToOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f, 3.0f, -4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = sum_to(x, target_shape);

    EXPECT_NEAR(result.item<float>(), 3.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, IdentityProperty)
{
    // 测试恒等性质：sum_to(x, x.shape()) = x
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sum_to(x, x.shape());

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, AssociativeProperty)
{
    // 测试结合性质：sum_to(sum_to(x, shape1), shape2) = sum_to(x, shape2)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape shape1{2, 1};
    Shape shape2{1, 1};

    auto result1 = sum_to(sum_to(x, shape1), shape2);
    auto result2 = sum_to(x, shape2);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, CommutativeProperty)
{
    // 测试交换性质：sum_to(x + y, shape) = sum_to(x, shape) + sum_to(y, shape)
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result1 = sum_to(x + y, target_shape);
    auto result2 = sum_to(x, target_shape) + sum_to(y, target_shape);

    EXPECT_NEAR(result1.item<float>(), result2.item<float>(), origin::test::TestTolerance::kDefault);
}

// ==================== 更多测试用例（新增） ====================

TEST_P(SumToOperatorTest, ForwardMultipleAxes)
{
    // 测试沿多个轴求和
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 36.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardPartialReduction)
{
    // 测试部分归约：从 (2, 3) 到 (2, 1)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({6.0f, 15.0f}, Shape{2, 1}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0f, -2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 4.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SumToOperatorTest, ForwardHighPrecision)
{
    // 测试高精度值
    auto x = Tensor({1e-6f, 2e-6f, 3e-6f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = sum_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 6e-6f, 1e-8f);
}

TEST_P(SumToOperatorTest, BackwardPartialReduction)
{
    // 测试部分归约的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    Shape target_shape{2, 1};

    auto y = sum_to(x, target_shape);
    y.backward();

    // 梯度应该广播回原始形状
    EXPECT_EQ(x.grad().shape(), x.shape());
    auto expected_grad = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(SumToOperatorTest);
