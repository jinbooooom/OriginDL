#include <gtest/gtest.h>
#include <vector>
#include "origin.h"
#include "origin/operators/conv/conv2d.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;

/**
 * @brief 测试 transpose_conv_output_kernel 的正确性
 * @details 验证从 (N, OH, OW, OC) 到 (N, OC, OH, OW) 的转置是否正确
 */
class Conv2dTransposeOutputTest : public origin::test::OperatorTestBase
{
};

TEST_P(Conv2dTransposeOutputTest, TransposeOutputShape)
{
    // 创建一个简单的卷积测试
    // 输入: (1, 1, 4, 4) - 单个通道，4x4图像
    // 卷积核: (2, 1, 2, 2) - 2个输出通道，2x2卷积核
    // 期望输出: (1, 2, 3, 3)
    
    std::vector<float> x_data(16, 1.0f);  // 全1的4x4图像
    auto x = Tensor(x_data, Shape{1, 1, 4, 4}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> W_data(8, 1.0f);  // 全1的2x2x2卷积核
    auto W = Tensor(W_data, Shape{2, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 1, 0);

    // 验证输出形状: (1, 2, 3, 3)
    Shape expected_shape{1, 2, 3, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出值：每个输出位置应该是4（2x2卷积核，每个值都是1，所以2*2=4）
    // 但这里我们主要验证形状和转置的正确性
    auto result_data = result.to_cpu().data<float>();
    EXPECT_EQ(result_data.size(), 18U);  // 1 * 2 * 3 * 3 = 18

    // 打印前几个值用于调试
    if (deviceType() == DeviceType::kCUDA)
    {
        std::cout << "CUDA Conv2d Output (first 10 values): ";
        for (size_t i = 0; i < std::min(10UL, result_data.size()); ++i)
        {
            std::cout << result_data[i] << " ";
        }
        std::cout << std::endl;
    }
}

TEST_P(Conv2dTransposeOutputTest, TransposeOutputValues)
{
    // 创建一个更具体的测试，验证转置后的值是否正确
    // 输入: (1, 1, 3, 3)
    // 卷积核: (2, 1, 2, 2)
    // 输出: (1, 2, 2, 2)
    
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 
                                  4.0f, 5.0f, 6.0f, 
                                  7.0f, 8.0f, 9.0f};
    auto x = Tensor(x_data, Shape{1, 1, 3, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 第一个卷积核: 全1
    // 第二个卷积核: 全2
    std::vector<float> W_data = {1.0f, 1.0f, 1.0f, 1.0f,  // 第一个卷积核
                                  2.0f, 2.0f, 2.0f, 2.0f}; // 第二个卷积核
    auto W = Tensor(W_data, Shape{2, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = conv2d(x, W, nullptr, 1, 0);

    Shape expected_shape{1, 2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 手动计算期望结果：
    // 对于第一个卷积核（全1）：
    // 左上: 1*1 + 2*1 + 4*1 + 5*1 = 12
    // 右上: 2*1 + 3*1 + 5*1 + 6*1 = 16
    // 左下: 4*1 + 5*1 + 7*1 + 8*1 = 24
    // 右下: 5*1 + 6*1 + 8*1 + 9*1 = 28
    // 对于第二个卷积核（全2）：
    // 左上: 1*2 + 2*2 + 4*2 + 5*2 = 24
    // 右上: 2*2 + 3*2 + 5*2 + 6*2 = 32
    // 左下: 4*2 + 5*2 + 7*2 + 8*2 = 48
    // 右下: 5*2 + 6*2 + 8*2 + 9*2 = 56
    
    // 输出形状是 (1, 2, 2, 2)，即 [batch, channel, height, width]
    // 所以输出应该是: [12, 16, 24, 28, 24, 32, 48, 56]
    std::vector<float> expected_data = {12.0f, 16.0f, 24.0f, 28.0f,  // 第一个通道
                                        24.0f, 32.0f, 48.0f, 56.0f}; // 第二个通道
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

INSTANTIATE_TEST_SUITE_P(Conv2dTransposeOutput, Conv2dTransposeOutputTest,
                         ::testing::Values(DeviceType::kCPU, DeviceType::kCUDA),
                         origin::test::DeviceTestBase::TestName);

