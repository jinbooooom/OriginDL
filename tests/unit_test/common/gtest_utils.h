#ifndef __ORIGIN_DL_GTEST_UTILS_H__
#define __ORIGIN_DL_GTEST_UTILS_H__

#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include "test_utils.h"

namespace origin
{
namespace test
{

/**
 * @brief GoogleTest断言包装类
 * @details 提供更友好的张量比较断言，依赖GoogleTest框架
 */
class GTestUtils
{
public:
    /**
     * @brief 张量相等断言
     * @param a 第一个张量
     * @param b 第二个张量
     * @param tolerance 容忍度，默认使用TestTolerance::kDefault
     * @details 如果断言失败，会输出详细的错误信息
     */
    static void EXPECT_TENSORS_EQ(const Tensor &a, const Tensor &b, double tolerance = TestTolerance::kDefault);

    /**
     * @brief 张量近似相等断言（使用相对误差和绝对误差）
     * @param a 第一个张量
     * @param b 第二个张量
     * @param rtol 相对误差容忍度，默认1e-5
     * @param atol 绝对误差容忍度，默认1e-8
     */
    static void EXPECT_TENSORS_NEAR(const Tensor &a, const Tensor &b, double rtol = 1e-5, double atol = 1e-8);

    /**
     * @brief 张量形状断言
     * @param tensor 要检查的张量
     * @param expected_shape 期望的形状
     */
    static void EXPECT_TENSOR_SHAPE(const Tensor &tensor, const Shape &expected_shape);

    /**
     * @brief 张量设备断言
     * @param tensor 要检查的张量
     * @param expected_device 期望的设备类型
     */
    static void EXPECT_TENSOR_DEVICE(const Tensor &tensor, DeviceType expected_device);

    /**
     * @brief 张量设备断言（使用Device对象）
     * @param tensor 要检查的张量
     * @param expected_device 期望的设备
     */
    static void EXPECT_TENSOR_DEVICE(const Tensor &tensor, const Device &expected_device);
};

}  // namespace test
}  // namespace origin

#endif  // __ORIGIN_DL_GTEST_UTILS_H__
