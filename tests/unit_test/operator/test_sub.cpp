#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief 减法算子测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          无GPU环境只运行CPU测试，有GPU环境运行CPU+CUDA测试
 */
class SubOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(SubOperatorTest, ForwardBasic)
{
    // 测试基本减法运算
    auto x0 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({4.0f, 4.0f, 4.0f, 4.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x0 = Tensor({5.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = x0 - x1;

    EXPECT_EQ(result.shape(), Shape{2});
    auto expected = Tensor({4.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, ForwardScalarTensor)
{
    // 测试标量与张量的减法
    auto x       = Tensor({5.0f, 6.0f, 7.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    float scalar = 2.0f;

    auto result1 = x - scalar;
    auto result2 = scalar - x;

    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});

    auto expected1 = Tensor({3.0f, 4.0f, 5.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected2 = Tensor({-3.0f, -4.0f, -5.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, expected1, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result2, expected2, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, ForwardZeroTensor)
{
    // 测试零张量减法
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    // 结果应该等于x0
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x0, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, ForwardNegativeValues)
{
    // 测试负值减法
    auto x0 = Tensor({-1.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    auto expected = Tensor({-4.0f, -6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(SubOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::sub(x0, x1);
    y.backward();

    // 减法算子的梯度：∂y/∂x0 = 1, ∂y/∂x1 = -1
    auto expected_gx0 = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected_gx1 = Tensor({-1.0f, -1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_gx1, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x0 = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({1.0f, 1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::sub(x0, x1);
    y.backward();

    // 减法算子的梯度：∂y/∂x0 = 1, ∂y/∂x1 = -1
    auto expected_gx0 = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected_gx1 = Tensor({-1.0f, -1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_gx1, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量减法反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::sub(x0, x1);
    y.backward();

    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector<float>();
    auto gx1_data = x1.grad().to_vector<float>();

    EXPECT_EQ(gx0_data.size(), 2U);
    EXPECT_EQ(gx1_data.size(), 1U);

    // x0的梯度应该是全1
    auto expected_gx0 = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);

    // x1的梯度应该是-2（广播后的梯度）
    EXPECT_NEAR(gx1_data[0], -2.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(SubOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x0 = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 2.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data1(100, 5.0f);
    std::vector<float> data2(100, 2.0f);
    auto x0 = Tensor(data1, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor(data2, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    auto expected =
        Tensor(std::vector<float>(100, 3.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                     dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, Shape{2, 2, 2},
                     dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 期望值：x0[i] - x1[i]
    auto expected = Tensor({0.9f, 1.8f, 2.7f, 3.6f, 4.5f, 5.4f, 6.3f, 7.2f}, Shape{2, 2, 2},
                           dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(SubOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x0 = Tensor({1e10f, 1e-10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1e-10f, 1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    auto expected = Tensor({1e10f, -1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e-3);  // 使用更宽松的容忍度
}

TEST_P(SubOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x0 = Tensor({0.3f, 0.4f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::sub(x0, x1);

    auto expected = Tensor({0.2f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 原地操作测试 ====================

TEST_P(SubOperatorTest, InplaceBasic)
{
    // 测试基本原地减法运算
    auto x0 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    F::sub_(x0, x1);

    auto expected = Tensor({4.0f, 4.0f, 4.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SubOperatorTest, InplaceZeroTensor)
{
    // 测试零张量原地减法
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto x0_original = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    F::sub_(x0, x1);

    // 结果应该等于x0的原始值
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, x0_original, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(SubOperatorTest);
