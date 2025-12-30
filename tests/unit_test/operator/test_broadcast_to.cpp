#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;
/**
 * @brief broadcast_to算子测试类（参数化版本）
 */
class BroadcastToOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(BroadcastToOperatorTest, ForwardBasic)
{
    // 测试基本广播运算（匹配libtorch的行主序行为）
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    // libtorch行主序：[1,2] expand到[2,2] = [[1,2],[1,2]] = [1,2,1,2]
    auto expected = Tensor({1.0f, 2.0f, 1.0f, 2.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, ForwardScalar)
{
    // 测试标量广播
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({5.0f, 5.0f, 5.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, ForwardToSameShape)
{
    // 测试相同形状（应该不变）
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, ForwardToLargerShape)
{
    // 测试到更大形状
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, ForwardTo3D)
{
    // 测试到三维形状
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 3, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();
    
    // 验证广播的正确性
    for (size_t i = 0; i < result_data.size(); i += 2)
    {
        EXPECT_NEAR(result_data[i], 1.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(result_data[i + 1], 2.0f, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(BroadcastToOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor::zeros(target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(BroadcastToOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    // broadcast_to算子的梯度：∂y/∂x = sum_to(gy, x.shape())
    auto expected_grad = Tensor({2.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor({2.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, BackwardToSameShape)
{
    // 测试相同形状的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, BackwardToLargerShape)
{
    // 测试到更大形状的反向传播
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();
    EXPECT_EQ(gx_data.size(), 1U);
    EXPECT_NEAR(gx_data[0], 3.0f, origin::test::TestTolerance::kDefault);  // 广播的梯度应该求和
}

// ==================== 边界情况测试 ====================

TEST_P(BroadcastToOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{1};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(10, 1.0f);
    auto x = Tensor(data, Shape{10}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{10, 10};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();
    
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0f, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(BroadcastToOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); i += 2)
    {
        EXPECT_NEAR(result_data[i], 1.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(result_data[i + 1], 2.0f, origin::test::TestTolerance::kDefault);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_P(BroadcastToOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10f, 1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1e10f, 1e-10f, 1e10f, 1e-10f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e7);
}

TEST_P(BroadcastToOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({0.1f, 0.2f, 0.1f, 0.2f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(BroadcastToOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); i += 2)
    {
        EXPECT_NEAR(result_data[i], 1.0f, origin::test::TestTolerance::kDefault);
        EXPECT_NEAR(result_data[i + 1], -2.0f, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(BroadcastToOperatorTest, IdentityProperty)
{
    // 测试恒等性质：broadcast_to(x, x.shape()) = x
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = broadcast_to(x, x.shape());

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, AssociativeProperty)
{
    // 测试结合性质：broadcast_to(broadcast_to(x, shape1), shape2) = broadcast_to(x, shape2)
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape shape1{2, 2};
    Shape shape2{2, 2, 2};

    auto result1 = broadcast_to(broadcast_to(x, shape1), shape2);
    auto result2 = broadcast_to(x, shape2);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, CommutativeProperty)
{
    // 测试交换性质：broadcast_to(x + y, shape) = broadcast_to(x, shape) + broadcast_to(y, shape)
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result1 = broadcast_to(x + y, target_shape);
    auto result2 = broadcast_to(x, target_shape) + broadcast_to(y, target_shape);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

// ==================== 维度感知广播测试（新增） ====================

TEST_P(BroadcastToOperatorTest, DimensionAwareBroadcast2DTo2D)
{
    // 测试维度感知广播：从 (2, 1) 到 (2, 2)
    // 应该得到 [[1, 1], [2, 2]] = [1, 1, 2, 2]
    auto x = Tensor({1.0f, 2.0f}, Shape{2, 1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 1.0f, 2.0f, 2.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, DimensionAwareBroadcast1DTo2D)
{
    // 测试维度感知广播：从 (3,) 到 (2, 3)
    // 应该得到 [[1, 2, 3], [1, 2, 3]] = [1, 2, 3, 1, 2, 3]
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 3};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(BroadcastToOperatorTest, DimensionAwareBroadcast3D)
{
    // 测试维度感知广播：从 (1, 2, 1) 到 (3, 2, 4)
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 2, 1}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{3, 2, 4};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    // 验证每个 (2,) 切片都是 [1, 1, 1, 1] 或 [2, 2, 2, 2]
    auto result_data = result.to_vector<float>();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            float expected_val = (j == 0) ? 1.0f : 2.0f;
            for (int k = 0; k < 4; ++k)
            {
                size_t idx = i * 8 + j * 4 + k;
                EXPECT_NEAR(result_data[idx], expected_val, origin::test::TestTolerance::kDefault);
            }
        }
    }
}

TEST_P(BroadcastToOperatorTest, DimensionAwareBroadcastComplex)
{
    // 测试复杂维度感知广播：从 (1, 3) 到 (2, 3)
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    Shape target_shape{2, 3};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto expected = Tensor({1.0f, 2.0f, 3.0f, 1.0f, 2.0f, 3.0f}, target_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(BroadcastToOperatorTest);
