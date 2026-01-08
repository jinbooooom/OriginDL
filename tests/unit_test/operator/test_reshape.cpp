#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
/**
 * @brief reshape算子测试类（参数化版本）
 */
class ReshapeOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(ReshapeOperatorTest, ForwardBasic)
{
    // 测试基本重塑运算
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4, 1};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ForwardToSameShape)
{
    // 测试相同形状（应该不变）
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = reshape(x, target_shape);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ForwardTo1D)
{
    // 测试重塑为一维
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{6};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected =
        Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ForwardTo2D)
{
    // 测试重塑为二维
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{6}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 3};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected =
        Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ForwardTo3D)
{
    // 测试重塑为三维
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{8},
                    dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2, 2};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, target_shape,
                           dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor::zeros(target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(ReshapeOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4, 1};

    auto y = reshape(x, target_shape);
    y.backward();

    // 重塑算子的梯度：∂y/∂x = reshape(gy, x.shape())
    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4, 1};

    auto y = reshape(x, target_shape);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, BackwardToSameShape)
{
    // 测试相同形状的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto y = reshape(x, target_shape);
    y.backward();

    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, BackwardToDifferentShape)
{
    // 测试不同形状的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3, 2};

    auto y = reshape(x, target_shape);
    y.backward();

    auto expected_grad = Tensor::ones(Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(ReshapeOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1, 1};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 1.0f);
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{100};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor(std::vector<float>(100, 1.0f), target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4, 2};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, target_shape,
                           dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(ReshapeOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10f, 1e-10f, 1e10f, 1e-10f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected =
        Tensor({1e10f, 1e-10f, 1e10f, 1e-10f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e7);
}

TEST_P(ReshapeOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({0.1f, 0.2f, 0.3f, 0.4f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(ReshapeOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f, 3.0f, -4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, -2.0f, 3.0f, -4.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, IdentityProperty)
{
    // 测试恒等性质：reshape(x, x.shape()) = x
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = reshape(x, x.shape());

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, AssociativeProperty)
{
    // 测试结合性质：reshape(reshape(x, shape1), shape2) = reshape(x, shape2)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape shape1{3, 2};
    Shape shape2{6};

    auto result1 = reshape(reshape(x, shape1), shape2);
    auto result2 = reshape(x, shape2);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, CommutativeProperty)
{
    // 测试交换性质：reshape(x + y, shape) = reshape(x, shape) + reshape(y, shape)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{4};

    auto result1 = reshape(x + y, target_shape);
    auto result2 = reshape(x, target_shape) + reshape(y, target_shape);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(ReshapeOperatorTest, ElementCountValidation)
{
    // 测试元素数量验证
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 有效重塑
    Shape valid_shape{4};
    EXPECT_NO_THROW(reshape(x, valid_shape));

    // 无效重塑（元素数量不匹配）
    Shape invalid_shape{5};
    EXPECT_THROW(reshape(x, invalid_shape), std::exception);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(ReshapeOperatorTest);
