#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/pooling/avg_pool2d.h"

using namespace origin;

/**
 * @brief AvgPool2d 算子测试类（参数化版本）
 */
class AvgPool2dOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(AvgPool2dOperatorTest, ForwardBasic)
{
    // 测试基本平均池化操作
    // 输入: (1, 1, 4, 4) - 单个通道，4x4图像
    // 池化核: 2x2, stride=2, pad=0
    // 期望输出: (1, 1, 2, 2)

    std::vector<float> x_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 4, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = avg_pool2d(x, 2, 2, 0);

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 手动计算期望结果：
    // 左上: (1+2+5+6)/4 = 3.5
    // 右上: (3+4+7+8)/4 = 5.5
    // 左下: (9+10+13+14)/4 = 11.5
    // 右下: (11+12+15+16)/4 = 13.5
    std::vector<float> expected_data = {3.5f, 5.5f, 11.5f, 13.5f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AvgPool2dOperatorTest, ForwardStride)
{
    // 测试带步长的池化
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 3, 3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = avg_pool2d(x, std::make_pair(2, 2), std::make_pair(1, 1), std::make_pair(0, 0));

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 1U);
    EXPECT_EQ(result.shape()[1], 1U);
    EXPECT_EQ(result.shape()[2], 2U);
    EXPECT_EQ(result.shape()[3], 2U);
}

TEST_P(AvgPool2dOperatorTest, ForwardPad)
{
    // 测试带填充的池化
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = avg_pool2d(x, 2, 1, 1);

    Shape expected_shape{1, 1, 3, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 1U);
    EXPECT_EQ(result.shape()[1], 1U);
    EXPECT_EQ(result.shape()[2], 3U);
    EXPECT_EQ(result.shape()[3], 3U);
}

TEST_P(AvgPool2dOperatorTest, ForwardSingleElement)
{
    // 测试单元素池化
    std::vector<float> x_data = {5.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = avg_pool2d(x, 1, 1, 0);

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(AvgPool2dOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = avg_pool2d(x, 2, 2, 0);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度不为零
    auto gx_data = x.grad().to_vector<float>();
    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NE(gx_data[i], 0.0f);
    }
}

TEST_P(AvgPool2dOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    // 使用2x2输入，2x2池化，stride=2，这样所有输入元素都会被池化覆盖
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = avg_pool2d(x, 2, 2, 0);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度计算正确
    // 对于2x2池化，stride=2，只有一个输出元素，对应4个输入元素，梯度应该平均分配
    auto gx_data = x.grad().to_vector<float>();
    // 所有输入元素梯度应该相等（因为输出梯度默认为1）
    float expected_grad = 1.0f / 4.0f;  // 每个输入元素接收1/4的梯度
    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], expected_grad, origin::test::TestTolerance::kDefault);
    }
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(AvgPool2dOperatorTest);
