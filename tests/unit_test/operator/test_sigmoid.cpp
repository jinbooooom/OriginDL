#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"
#include "origin.h"

using namespace origin;

/**
 * @brief Sigmoid 算子测试类（参数化版本）
 */
class SigmoidOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(SigmoidOperatorTest, ForwardBasic)
{
    // 测试基本 Sigmoid 运算
    // sigmoid(0) = 0.5
    // sigmoid(1) ≈ 0.731
    // sigmoid(-1) ≈ 0.269
    auto x = Tensor({0.0f, 1.0f, -1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sigmoid(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {0.5f, 1.0f / (1.0f + std::exp(-1.0f)), 1.0f / (1.0f + std::exp(1.0f))};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SigmoidOperatorTest, ForwardZero)
{
    // 测试零值：sigmoid(0) = 0.5
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sigmoid(x);

    auto expected = Tensor::full(Shape{2}, 0.5f, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SigmoidOperatorTest, ForwardExtremeValues)
{
    // 测试极值
    // sigmoid(10) ≈ 0.9999
    // sigmoid(-10) ≈ 0.0001
    auto x = Tensor({10.0f, -10.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sigmoid(x);

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {1.0f / (1.0f + std::exp(-10.0f)), 1.0f / (1.0f + std::exp(10.0f))};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);

    // 验证极值特性：大正数接近1，大负数接近0
    auto result_data = result.to_vector<float>();
    EXPECT_GT(result_data[0], 0.999f);
    EXPECT_LT(result_data[1], 0.001f);
}

TEST_P(SigmoidOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量
    auto x = Tensor({0.0f, 1.0f, -1.0f, 2.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sigmoid(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {0.5f, 1.0f / (1.0f + std::exp(-1.0f)), 1.0f / (1.0f + std::exp(1.0f)),
                                        1.0f / (1.0f + std::exp(-2.0f))};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SigmoidOperatorTest, ForwardThreeDimensional)
{
    // 测试三维张量
    std::vector<float> input_data = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 3.0f, -3.0f, 0.5f};
    auto x                        = Tensor(input_data, Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sigmoid(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data;
    for (float val : input_data)
    {
        expected_data.push_back(1.0f / (1.0f + std::exp(-val)));
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(SigmoidOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    auto x = Tensor({0.0f, 1.0f, -1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sigmoid(x);
    y.backward();

    // 计算期望梯度
    std::vector<float> expected_grad_data;
    for (float val : x.to_vector<float>())
    {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
        expected_grad_data.push_back(sigmoid_val * (1.0f - sigmoid_val));
    }
    auto expected_grad = Tensor(expected_grad_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SigmoidOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    // 注意：y.backward() 会自动使用全1的梯度，所以这里我们验证基本行为
    auto x = Tensor({0.0f, 1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sigmoid(x);
    y.backward();

    // 计算期望梯度（当gy=1时）
    std::vector<float> x_data = x.to_vector<float>();
    std::vector<float> expected_grad_data;
    for (float val : x_data)
    {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
        expected_grad_data.push_back(sigmoid_val * (1.0f - sigmoid_val));
    }
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SigmoidOperatorTest, BackwardZero)
{
    // 测试零值的梯度：sigmoid'(0) = 0.5 * 0.5 = 0.25
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = sigmoid(x);
    y.backward();

    auto expected_grad = Tensor::full(Shape{2}, 0.25f, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(SigmoidOperatorTest, RangeProperty)
{
    // 测试范围性质：sigmoid(x) 应该在 [0, 1] 范围内
    auto x = Tensor({-100.0f, -10.0f, 0.0f, 10.0f, 100.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = sigmoid(x);

    auto result_data = result.to_vector<float>();
    // 所有值应该在 [0, 1] 范围内
    for (float val : result_data)
    {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }

    // 对于非极值，应该在 (0, 1) 范围内
    // result_data[0] 是 sigmoid(-100)，可能接近0，允许等于0
    // result_data[1] 是 sigmoid(-10)，应该在 (0, 1) 范围内
    EXPECT_GT(result_data[1], 0.0f);  // sigmoid(-10) > 0
    EXPECT_LT(result_data[1], 1.0f);  // sigmoid(-10) < 1
    EXPECT_GT(result_data[2], 0.0f);  // sigmoid(0) = 0.5 > 0
    EXPECT_LT(result_data[2], 1.0f);  // sigmoid(0) = 0.5 < 1
    EXPECT_GT(result_data[3], 0.0f);  // sigmoid(10) > 0
    EXPECT_LT(result_data[3], 1.0f);  // sigmoid(10) < 1
    // result_data[4] 是 sigmoid(100)，可能接近1，允许等于1
}

TEST_P(SigmoidOperatorTest, SymmetryProperty)
{
    // 测试对称性质：sigmoid(-x) = 1 - sigmoid(x)
    auto x     = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto neg_x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto sigmoid_x     = sigmoid(x);
    auto sigmoid_neg_x = sigmoid(neg_x);
    auto ones          = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected      = ones - sigmoid_x;

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(sigmoid_neg_x, expected, origin::test::TestTolerance::kDefault);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SigmoidOperatorTest);
