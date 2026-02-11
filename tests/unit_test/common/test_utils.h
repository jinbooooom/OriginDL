#ifndef __ORIGIN_DL_TEST_UTILS_H__
#define __ORIGIN_DL_TEST_UTILS_H__

#include <cmath>
#include <vector>
#include "origin.h"

namespace origin
{
namespace test
{

/**
 * @brief 精度容忍常量
 * @details 提供不同场景下的精度容忍度
 */
struct TestTolerance
{
    static constexpr double kDefault = 1e-3;  // 默认精度
    static constexpr double kStrict  = 1e-5;  // 严格精度
    static constexpr double kLoose   = 1e-2;  // 宽松精度
};

/**
 * @brief 通用测试工具类（不依赖测试框架）
 * @details 可用于单元测试、对比测试、示例代码等场景
 *          不依赖GoogleTest，提供纯工具函数
 */
class TestUtils
{
public:
    /**
     * @brief 比较两个浮点数是否相等（考虑浮点精度）
     * @param a 第一个浮点数
     * @param b 第二个浮点数
     * @param tolerance 容忍度，默认使用TestTolerance::kDefault
     * @return 如果 |a - b| < tolerance 返回true，否则返回false
     */
    static bool isEqual(double a, double b, double tolerance = TestTolerance::kDefault);

    /**
     * @brief 比较两个张量是否相等
     * @param a 第一个张量
     * @param b 第二个张量
     * @param tolerance 容忍度，默认使用TestTolerance::kDefault
     * @return 如果形状相同且所有元素在容忍度范围内相等返回true，否则返回false
     */
    static bool tensorsEqual(const Tensor &a, const Tensor &b, double tolerance = TestTolerance::kDefault);

    /**
     * @brief 比较两个张量是否近似相等（使用相对误差和绝对误差）
     * @param a 第一个张量
     * @param b 第二个张量
     * @param rtol 相对误差容忍度，默认1e-5
     * @param atol 绝对误差容忍度，默认1e-8
     * @return 如果形状相同且所有元素满足 |a - b| <= atol + rtol * max(|a|, |b|) 返回true
     */
    static bool tensorsNear(const Tensor &a, const Tensor &b, double rtol = 1e-5, double atol = 1e-8);

    /**
     * @brief 检查CUDA是否可用
     * @return 如果CUDA可用返回true，否则返回false
     */
    static bool isCudaAvailable();
};

}  // namespace test
}  // namespace origin

#endif  // __ORIGIN_DL_TEST_UTILS_H__
