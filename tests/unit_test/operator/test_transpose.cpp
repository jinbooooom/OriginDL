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
 * @brief 转置算子测试类（参数化版本）
 */
class TransposeOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(TransposeOperatorTest, ForwardBasic)
{
    // 测试基本转置运算,行主序
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 行主序：[[1,2],[3,4]]转置为[[1,3],[2,4]]，展开为[1,3,2,4]
    auto expected = Tensor({1.0f, 3.0f, 2.0f, 4.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, Forward3x2Matrix)
{
    // 测试3x2矩阵转置
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 3};
    EXPECT_EQ(result.shape(), expected_shape);
    // 行主序：[[1,2],[3,4],[5,6]]转置为[[1,3,5],[2,4,6]]，展开为[1,3,5,2,4,6]
    auto expected =
        Tensor({1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, ForwardSquareMatrix)
{
    // 测试方阵转置
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f}, Shape{3, 3},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{3, 3};
    EXPECT_EQ(result.shape(), expected_shape);
    // 行主序：[[1,2,3],[4,5,6],[7,8,9]]转置为[[1,4,7],[2,5,8],[3,6,9]]，展开为[1,4,7,2,5,8,3,6,9]
    auto expected = Tensor({1.0f, 4.0f, 7.0f, 2.0f, 5.0f, 8.0f, 3.0f, 6.0f, 9.0f}, expected_shape,
                           dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, ForwardOneDimensional)
{
    // 测试一维张量
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor::zeros(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor::zeros(expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(TransposeOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::transpose(x);
    y.backward();

    // 转置算子的梯度：∂y/∂x = F::transpose(gy)
    // 梯度应该是转置后的结果（gy=1时，梯度=1）
    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::transpose(x);
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, Backward3x2Matrix)
{
    // 测试3x2矩阵的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::transpose(x);
    y.backward();

    // 梯度应该是转置后的结果（gy=1时，梯度=1）
    auto expected_grad = Tensor::ones(Shape{3, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, BackwardOneDimensional)
{
    // 测试一维张量的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = F::transpose(x);
    y.backward();

    auto expected_grad = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(TransposeOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, LargeMatrix)
{
    // 测试大矩阵
    std::vector<float> data(100, 1.0f);
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected =
        Tensor(std::vector<float>(100, 1.0f), expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, ThreeDimensional)
{
    // 测试三维张量（应该只转置最后两个维度）
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector<float>();

    // 验证转置的正确性（行主序行为）
    // 原始：[[[1,2],[3,4]], [[5,6],[7,8]]] -> 转置：[[[1,3],[2,4]], [[5,7],[6,8]]]
    EXPECT_NEAR(result_data[0], 1.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[1], 3.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[2], 2.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[3], 4.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[4], 5.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[5], 7.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[6], 6.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(result_data[7], 8.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(TransposeOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10f, 1e-10f, 1e10f, 1e-10f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 行主序：[[1e10,1e-10],[1e10,1e-10]]转置为[[1e10,1e10],[1e-10,1e-10]]，展开为[1e10,1e10,1e-10,1e-10]
    auto expected =
        Tensor({1e10f, 1e10f, 1e-10f, 1e-10f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e7);
}

TEST_P(TransposeOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 行主序：[[0.1,0.2],[0.3,0.4]]转置为[[0.1,0.3],[0.2,0.4]]，展开为[0.1,0.3,0.2,0.4]
    auto expected = Tensor({0.1f, 0.3f, 0.2f, 0.4f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(TransposeOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f, 3.0f, -4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 行主序：[[1,-2],[3,-4]]转置为[[1,3],[-2,-4]]，展开为[1,3,-2,-4]
    auto expected = Tensor({1.0f, 3.0f, -2.0f, -4.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, IdentityProperty)
{
    // 测试恒等性质：F::transpose(F::transpose(x)) = x
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::transpose(F::transpose(x));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, CommutativeProperty)
{
    // 测试交换性质：F::transpose(x + y) = F::transpose(x) + F::transpose(y)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result1 = F::transpose(x + y);
    auto result2 = F::transpose(x) + F::transpose(y);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(TransposeOperatorTest, AssociativeProperty)
{
    // 测试结合性质：F::transpose(x * y) = F::transpose(y) * F::transpose(x)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result1 = F::transpose(x * y);
    auto result2 = F::transpose(y) * F::transpose(x);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(TransposeOperatorTest);
