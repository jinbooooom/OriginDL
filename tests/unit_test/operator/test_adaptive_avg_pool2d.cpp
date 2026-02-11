#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief AdaptiveAvgPool2d 算子测试类（参数化版本）
 */
class AdaptiveAvgPool2dOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(AdaptiveAvgPool2dOperatorTest, ForwardBasic)
{
    // 测试基本自适应平均池化操作
    // 输入: (1, 1, 4, 4) - 单个通道，4x4图像
    // 输出尺寸: (2, 2)
    // 期望输出: (1, 1, 2, 2)

    std::vector<float> x_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 4, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::adaptive_avg_pool2d(x, {2, 2});

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

TEST_P(AdaptiveAvgPool2dOperatorTest, ForwardToOne)
{
    // 测试输出为 1x1 的情况
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::adaptive_avg_pool2d(x, {1, 1});

    Shape expected_shape{1, 1, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);

    // 所有元素的平均值: (1+2+3+4)/4 = 2.5
    std::vector<float> expected_data = {2.5f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AdaptiveAvgPool2dOperatorTest, ForwardMultiChannel)
{
    // 测试多通道情况
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    auto x                    = Tensor(x_data, Shape{1, 2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::adaptive_avg_pool2d(x, {1, 1});

    Shape expected_shape{1, 2, 1, 1};
    EXPECT_EQ(result.shape(), expected_shape);

    // 通道0: (1+2+3+4)/4 = 2.5
    // 通道1: (5+6+7+8)/4 = 6.5
    std::vector<float> expected_data = {2.5f, 6.5f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AdaptiveAvgPool2dOperatorTest, ForwardNonDivisible)
{
    // 测试输入尺寸不能被输出尺寸整除的情况
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 3, 3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::adaptive_avg_pool2d(x, {2, 2});

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
}

INSTANTIATE_DEVICE_TEST_SUITE_P(AdaptiveAvgPool2dOperatorTest);
