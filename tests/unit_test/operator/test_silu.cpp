#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "origin/operators/activation/silu.h"
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief SiLU 算子测试类（参数化版本）
 */
class SiLUOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(SiLUOperatorTest, ForwardBasic)
{
    // 测试基本 SiLU 运算
    // SiLU(x) = x * F::sigmoid(x)
    // SiLU(0) = 0 * F::sigmoid(0) = 0 * 0.5 = 0
    // SiLU(1) = 1 * F::sigmoid(1) ≈ 1 * 0.731 = 0.731
    // SiLU(-1) = -1 * F::sigmoid(-1) ≈ -1 * 0.269 = -0.269
    auto x = Tensor({0.0f, 1.0f, -1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 计算期望值
    float sigmoid_0 = 0.5f;
    float sigmoid_1 = 1.0f / (1.0f + std::exp(-1.0f));
    float sigmoid_neg1 = 1.0f / (1.0f + std::exp(1.0f));
    
    std::vector<float> expected_data = {
        0.0f * sigmoid_0,      // 0
        1.0f * sigmoid_1,     // ≈ 0.731
        -1.0f * sigmoid_neg1  // ≈ -0.269
    };
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SiLUOperatorTest, ForwardZero)
{
    // 测试零值：SiLU(0) = 0 * F::sigmoid(0) = 0
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    auto expected = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SiLUOperatorTest, ForwardPositiveValues)
{
    // 测试正数
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 计算期望值
    std::vector<float> expected_data;
    for (float val : x.to_vector<float>())
    {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
        expected_data.push_back(val * sigmoid_val);
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SiLUOperatorTest, ForwardNegativeValues)
{
    // 测试负数
    auto x = Tensor({-1.0f, -2.0f, -3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 计算期望值
    std::vector<float> expected_data;
    for (float val : x.to_vector<float>())
    {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
        expected_data.push_back(val * sigmoid_val);
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(SiLUOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量
    auto x = Tensor({0.0f, 1.0f, -1.0f, 2.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data;
    for (float val : x.to_vector<float>())
    {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
        expected_data.push_back(val * sigmoid_val);
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(SiLUOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    // SiLU'(x) = F::sigmoid(x) * (1 + x * (1 - F::sigmoid(x)))
    auto x = Tensor({0.0f, 1.0f, -1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::silu(x);
    y.backward();

    // 计算期望梯度
    std::vector<float> expected_grad_data;
    for (float val : x.to_vector<float>())
    {
        float sigmoid_val = 1.0f / (1.0f + std::exp(-val));
        float grad = sigmoid_val * (1.0f + val * (1.0f - sigmoid_val));
        expected_grad_data.push_back(grad);
    }
    auto expected_grad = Tensor(expected_grad_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SiLUOperatorTest, BackwardZero)
{
    // 测试零值的梯度
    // SiLU'(0) = F::sigmoid(0) * (1 + 0 * (1 - F::sigmoid(0))) = 0.5 * 1 = 0.5
    auto x = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::silu(x);
    y.backward();

    auto expected_grad = Tensor::full(Shape{2}, 0.5f, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(SiLUOperatorTest, IdentityProperty)
{
    // 测试性质：对于大正数，SiLU(x) ≈ x
    auto x = Tensor({10.0f, 20.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    // 对于大正数，F::sigmoid(x) ≈ 1，所以 SiLU(x) ≈ x
    auto result_data = result.to_vector<float>();
    auto x_data = x.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], x_data[i], 0.1f);  // 允许一定误差
    }
}

TEST_P(SiLUOperatorTest, NegativeProperty)
{
    // 测试性质：对于大负数，SiLU(x) ≈ 0
    auto x = Tensor({-10.0f, -20.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::silu(x);

    // 对于大负数，F::sigmoid(x) ≈ 0，所以 SiLU(x) ≈ 0
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_NEAR(val, 0.0f, 0.1f);  // 允许一定误差
    }
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SiLUOperatorTest);

