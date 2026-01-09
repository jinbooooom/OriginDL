#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/pooling/max_pool2d.h"

using namespace origin;

/**
 * @brief MaxPool2d 算子测试类（参数化版本）
 */
class MaxPool2dOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(MaxPool2dOperatorTest, ForwardBasic)
{
    // 测试基本最大池化操作
    // 输入: (1, 1, 4, 4) - 单个通道，4x4图像
    // 池化核: 2x2, stride=2, pad=0
    // 期望输出: (1, 1, 2, 2)

    std::vector<float> x_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 4, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = max_pool2d(x, 2, 2, 0);

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 手动计算期望结果：
    // 左上: max(1,2,5,6) = 6
    // 右上: max(3,4,7,8) = 8
    // 左下: max(9,10,13,14) = 14
    // 右下: max(11,12,15,16) = 16
    std::vector<float> expected_data = {6.0f, 8.0f, 14.0f, 16.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(MaxPool2dOperatorTest, ForwardStride)
{
    // 测试带步长的池化
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 3, 3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = max_pool2d(x, std::make_pair(2, 2), std::make_pair(1, 1), std::make_pair(0, 0));

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 1U);
    EXPECT_EQ(result.shape()[1], 1U);
    EXPECT_EQ(result.shape()[2], 2U);
    EXPECT_EQ(result.shape()[3], 2U);
}

TEST_P(MaxPool2dOperatorTest, ForwardPad)
{
    // 测试带填充的池化
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = max_pool2d(x, 2, 1, 1);

    Shape expected_shape{1, 1, 3, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 1U);
    EXPECT_EQ(result.shape()[1], 1U);
    EXPECT_EQ(result.shape()[2], 3U);
    EXPECT_EQ(result.shape()[3], 3U);
}

TEST_P(MaxPool2dOperatorTest, ForwardSingleElement)
{
    // 测试单元素池化
    std::vector<float> x_data = {5.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = max_pool2d(x, 1, 1, 0);

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 5.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(MaxPool2dOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = max_pool2d(x, 2, 2, 0);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度不为零
    auto gx_data = x.grad().to_vector<float>();
    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        // 对于2x2池化，stride=2，只有一个输出元素，对应4个输入元素
        // 只有最大值位置的梯度不为0
        if (i == 3)  // 4.0f是最大值
        {
            EXPECT_NE(gx_data[i], 0.0f);
        }
        else
        {
            EXPECT_EQ(gx_data[i], 0.0f);
        }
    }
}

TEST_P(MaxPool2dOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    // 使用2x2输入，2x2池化，stride=2，这样所有输入元素都会被池化覆盖
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = max_pool2d(x, 2, 2, 0);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度计算正确
    // 对于2x2池化，stride=2，只有一个输出元素，对应4个输入元素
    // 只有最大值位置的梯度为1（默认输出梯度），其他为0
    auto gx_data = x.grad().to_vector<float>();
    // 4.0f是最大值，索引为3
    EXPECT_NEAR(gx_data[3], 1.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(gx_data[0], 0.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(gx_data[1], 0.0f, origin::test::TestTolerance::kDefault);
    EXPECT_NEAR(gx_data[2], 0.0f, origin::test::TestTolerance::kDefault);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(MaxPool2dOperatorTest);
