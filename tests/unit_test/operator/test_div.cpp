#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;
/**
 * @brief 除法算子测试类（参数化版本）
 */
class DivOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(DivOperatorTest, ForwardBasic)
{
    // 测试基本除法运算
    auto x0 = Tensor({6.0f, 8.0f, 10.0f, 12.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({3.0f, 2.0f, 2.0f, 2.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x0 = Tensor({6.0f, 8.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = x0 / x1;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({3.0f, 2.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, ForwardScalarTensor)
{
    // 测试标量与张量的除法
    auto x       = Tensor({6.0f, 8.0f, 10.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    float scalar = 2.0f;

    auto result1 = x / scalar;
    auto result2 = scalar / x;

    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});

    auto expected1 = Tensor({3.0f, 4.0f, 5.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected2 = Tensor({1.0f / 3.0f, 0.25f, 0.2f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, expected1, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result2, expected2, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, ForwardOneTensor)
{
    // 测试除以1的张量
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    // 结果应该等于x0
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x0, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, ForwardNegativeValues)
{
    // 测试负值除法
    auto x0 = Tensor({-6.0f, -8.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    auto expected = Tensor({-3.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(DivOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({6.0f, 8.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::div(x0, x1);
    y.backward();

    // 除法算子的梯度：∂y/∂x0 = 1/x1, ∂y/∂x1 = -x0/x1²
    auto x0_data = x0.to_vector<float>();
    auto x1_data = x1.to_vector<float>();

    auto expected_gx0 = Tensor({1.0f / 2.0f, 1.0f / 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected_gx1 = Tensor({-6.0f / 4.0f, -8.0f / 16.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_gx1, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x0 = Tensor({4.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::div(x0, x1);
    y.backward();

    auto x0_data = x0.to_vector<float>();
    auto x1_data = x1.to_vector<float>();

    // 梯度会累积，所以需要计算累积后的值
    auto expected_gx0 = Tensor({1.0f / 2.0f, 1.0f / 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected_gx1 = Tensor({-4.0f / 4.0f, -6.0f / 9.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_gx1, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量除法反向传播
    auto x0 = Tensor({6.0f, 8.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::div(x0, x1);
    y.backward();

    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector<float>();
    auto gx1_data = x1.grad().to_vector<float>();

    EXPECT_EQ(gx0_data.size(), 2U);
    EXPECT_EQ(gx1_data.size(), 1U);

    // x0的梯度应该是1/2 = 0.5
    auto expected_gx0 = Tensor({0.5f, 0.5f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);

    // x1的梯度应该是-(6+8)/4 = -14/4 = -3.5
    EXPECT_NEAR(gx1_data[0], -3.5f, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(DivOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x0 = Tensor({15.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data1(100, 6.0f);
    std::vector<float> data2(100, 2.0f);
    auto x0 = Tensor(data1, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor(data2, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    auto expected =
        Tensor(std::vector<float>(100, 3.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x0 = Tensor({2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f}, Shape{2, 2, 2},
                     dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                     dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    auto expected = Tensor(std::vector<float>(8, 2.0f), Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(DivOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x0 = Tensor({1e10f, 1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1e-10f, 1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    auto expected = Tensor({1e20f, 1e-20f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e-3);
}

TEST_P(DivOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x0 = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({0.2f, 0.4f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::div(x0, x1);

    auto expected = Tensor({0.5f, 0.5f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 原地操作测试 ====================

TEST_P(DivOperatorTest, InplaceBasic)
{
    // 测试基本原地除法运算
    auto x0 = Tensor({6.0f, 8.0f, 10.0f, 12.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    
    F::div_(x0, x1);
    
    auto expected = Tensor({3.0f, 2.0f, 2.0f, 2.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(DivOperatorTest, InplaceOneTensor)
{
    // 测试除以1的张量原地操作
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    
    auto x0_original = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    
    F::div_(x0, x1);
    
    // 结果应该等于x0的原始值
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, x0_original, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(DivOperatorTest);
