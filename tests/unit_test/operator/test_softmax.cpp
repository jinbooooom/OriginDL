#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"

using namespace origin;

/**
 * @brief Softmax 算子测试类（参数化版本）
 */
class SoftmaxOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(SoftmaxOperatorTest, ForwardBasic)
{
    // 测试基本 softmax 运算
    // 输入: [1.0, 2.0, 3.0]
    // 预期: 每个元素经过 softmax 归一化，和为 1
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 验证和为 1
    auto sum_result = sum(result);
    EXPECT_NEAR(sum_result.item<float>(), 1.0f, 1e-5f);
    
    // 验证所有值在 [0, 1] 范围内
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
}

TEST_P(SoftmaxOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量的 softmax（沿最后一个维度）
    // 输入: [[1.0, 2.0], [3.0, 4.0]]
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 验证每一行的和为 1
    auto result_data = result.to_vector<float>();
    float row1_sum = result_data[0] + result_data[1];
    float row2_sum = result_data[2] + result_data[3];
    EXPECT_NEAR(row1_sum, 1.0f, 1e-5f);
    EXPECT_NEAR(row2_sum, 1.0f, 1e-5f);
}

TEST_P(SoftmaxOperatorTest, ForwardLargeValues)
{
    // 测试大值情况（数值稳定性）
    // 输入: [100.0, 101.0, 102.0]
    // 应该能正确处理，不会溢出
    auto x = Tensor({100.0f, 101.0f, 102.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 验证和为 1
    auto sum_result = sum(result);
    EXPECT_NEAR(sum_result.item<float>(), 1.0f, 1e-4f);
    
    // 验证所有值都是有效的（不是 NaN 或 Inf）
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
    }
}

TEST_P(SoftmaxOperatorTest, ForwardNegativeValues)
{
    // 测试负值情况
    auto x = Tensor({-1.0f, 0.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 验证和为 1
    auto sum_result = sum(result);
    EXPECT_NEAR(sum_result.item<float>(), 1.0f, 1e-5f);
}

TEST_P(SoftmaxOperatorTest, ForwardEqualValues)
{
    // 测试所有值相等的情况
    auto x = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 所有值应该相等（约 1/3）
    auto result_data = result.to_vector<float>();
    float expected_val = 1.0f / 3.0f;
    for (float val : result_data)
    {
        EXPECT_NEAR(val, expected_val, 1e-5f);
    }
}

// ==================== 反向传播测试 ====================

TEST_P(SoftmaxOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = softmax(x);
    y.backward();

    // 验证梯度不为空（通过检查梯度形状）
    EXPECT_EQ(x.grad().shape(), x.shape());
}

TEST_P(SoftmaxOperatorTest, BackwardWithGradient)
{
    // 测试反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = softmax(x);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());
}

TEST_P(SoftmaxOperatorTest, BackwardTwoDimensional)
{
    // 测试 2D 张量的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = softmax(x);
    y.backward();

    // 验证梯度不为空（通过检查梯度形状）
    EXPECT_EQ(x.grad().shape(), x.shape());
}

// ==================== 边界情况测试 ====================

TEST_P(SoftmaxOperatorTest, SingleElement)
{
    // 测试单个元素
    auto x = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 1.0f, 1e-5f);  // 单个元素的 softmax 应该是 1
}

TEST_P(SoftmaxOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100);
    for (size_t i = 0; i < 100; ++i)
    {
        data[i] = static_cast<float>(i);
    }
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 验证每一行的和为 1
    auto result_data = result.to_vector<float>();
    for (int row = 0; row < 10; ++row)
    {
        float row_sum = 0.0f;
        for (int col = 0; col < 10; ++col)
        {
            row_sum += result_data[row * 10 + col];
        }
        EXPECT_NEAR(row_sum, 1.0f, 1e-4f);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_P(SoftmaxOperatorTest, NumericalStability)
{
    // 测试数值稳定性：非常大的值
    auto x = Tensor({1000.0f, 1001.0f, 1002.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = softmax(x);

    // 验证没有溢出（所有值应该是有效的）
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
    
    // 验证和为 1
    auto sum_result = sum(result);
    EXPECT_NEAR(sum_result.item<float>(), 1.0f, 1e-3f);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SoftmaxOperatorTest);

