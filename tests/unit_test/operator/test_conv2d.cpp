#include <gtest/gtest.h>
#include <vector>
#include "origin.h"
#include "origin/operators/conv/conv2d.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;

/**
 * @brief Conv2d 算子测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 */
class Conv2dOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(Conv2dOperatorTest, ForwardBasic)
{
    // 测试基本卷积操作
    // 输入: (1, 1, 3, 3) - 单个通道，3x3图像
    // 卷积核: (1, 1, 2, 2) - 单个输出通道，2x2卷积核
    // 期望输出: (1, 1, 2, 2)

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto x = Tensor(x_data, Shape{1, 1, 3, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 1, 0);

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 手动计算期望结果：
    // 左上: 1*1 + 2*1 + 4*1 + 5*1 = 12
    // 右上: 2*1 + 3*1 + 5*1 + 6*1 = 16
    // 左下: 4*1 + 5*1 + 7*1 + 8*1 = 24
    // 右下: 5*1 + 6*1 + 8*1 + 9*1 = 28
    std::vector<float> expected_data = {12.0f, 16.0f, 24.0f, 28.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(Conv2dOperatorTest, ForwardWithStride)
{
    // 测试带步长的卷积
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto x = Tensor(x_data, Shape{1, 1, 3, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 2, 0);

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);

    // stride=2, 只有左上角一个位置
    // 左上: 1*1 + 2*1 + 4*1 + 5*1 = 12
    std::vector<float> expected_data = {12.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(Conv2dOperatorTest, ForwardWithPadding)
{
    // 测试带填充的卷积
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 1, 1);

    Shape expected_shape{1, 1, 3, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 1U);
    EXPECT_EQ(result.shape()[1], 1U);
    EXPECT_EQ(result.shape()[2], 3U);
    EXPECT_EQ(result.shape()[3], 3U);
}

TEST_P(Conv2dOperatorTest, ForwardWithBias)
{
    // 测试带偏置的卷积
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> b_data = {1.0f};
    auto b = Tensor(b_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, &b, 1, 0);

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);

    // 结果应该是卷积结果 + 偏置
    // 卷积: 1*1 + 2*1 + 3*1 + 4*1 = 10
    // 加上偏置: 10 + 1 = 11
    std::vector<float> expected_data = {11.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(Conv2dOperatorTest, ForwardMultipleChannels)
{
    // 测试多通道卷积
    // 输入: (1, 2, 2, 2) - 2个通道，2x2图像
    // 卷积核: (1, 2, 2, 2) - 1个输出通道，2个输入通道，2x2卷积核
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto x = Tensor(x_data, Shape{1, 2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 1, 0);

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 1U);
    EXPECT_EQ(result.shape()[1], 1U);
    EXPECT_EQ(result.shape()[2], 1U);
    EXPECT_EQ(result.shape()[3], 1U);
}

// ==================== 反向传播测试 ====================

TEST_P(Conv2dOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = conv2d(x, W, nullptr, 1, 0);
    y.backward();

    // 验证梯度不为零
    auto gx_data = x.grad().to_vector<float>();
    auto gW_data = W.grad().to_vector<float>();

    EXPECT_EQ(gx_data.size(), 4U);
    EXPECT_EQ(gW_data.size(), 4U);

    // 验证梯度计算正确（至少不为零）
    bool gx_nonzero = false;
    bool gW_nonzero = false;
    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        if (std::abs(gx_data[i]) > 1e-6f)
        {
            gx_nonzero = true;
            break;
        }
    }
    for (size_t i = 0; i < gW_data.size(); ++i)
    {
        if (std::abs(gW_data[i]) > 1e-6f)
        {
            gW_nonzero = true;
            break;
        }
    }

    EXPECT_TRUE(gx_nonzero);
    EXPECT_TRUE(gW_nonzero);
}

TEST_P(Conv2dOperatorTest, BackwardWithBias)
{
    // 测试带偏置的反向传播
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> b_data = {1.0f};
    auto b = Tensor(b_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = conv2d(x, W, &b, 1, 0);
    y.backward();

    // 验证偏置梯度不为零
    auto gb_data = b.grad().to_vector<float>();
    EXPECT_EQ(gb_data.size(), 1U);
    EXPECT_NE(gb_data[0], 0.0f);
}

// ==================== 边界情况测试 ====================

TEST_P(Conv2dOperatorTest, SingleElementInput)
{
    // 测试单元素输入
    std::vector<float> x_data = {1.0f};
    auto x = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data = {1.0f};
    auto W = Tensor(W_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 1, 0);

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {1.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(Conv2dOperatorTest);

