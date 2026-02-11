#ifndef __ORIGIN_DL_DEVICE_TEST_BASE_H__
#define __ORIGIN_DL_DEVICE_TEST_BASE_H__

#include <gtest/gtest.h>
#include <vector>
#include "gtest_utils.h"
#include "test_utils.h"

namespace origin
{
namespace test
{

/**
 * @brief 设备感知的测试基类（参数化版本）
 * @details 使用GoogleTest的参数化测试，自动为CPU和CUDA生成测试用例
 *          使用origindl已有的DeviceType枚举，不重新定义
 *
 * 工作原理：
 *
 * 1. 参数生成阶段：
 *    - INSTANTIATE_DEVICE_TEST_SUITE_P宏调用GetAvailableDevices()获取可用设备列表
 *    - 例如：{DeviceType::kCPU, DeviceType::kCUDA}（如果CUDA可用）
 *    - GoogleTest使用ValuesIn()为每个设备类型生成一个测试参数
 *
 * 2. 测试实例生成：
 *    - 对于每个TEST_P测试，GoogleTest会为每个设备类型创建一个测试实例
 *    - 例如：TEST_P(AddOperatorTest, ForwardBasic) 会生成：
 *      * AllDevices/AddOperatorTest.ForwardBasic/CPU
 *      * AllDevices/AddOperatorTest.ForwardBasic/CUDA（如果CUDA可用）
 *
 * 3. 测试执行阶段：
 *    - SetUp()中通过GetParam()获取当前测试实例的设备类型
 *    - device_type_ = GetParam() 从GoogleTest参数化测试获取设备类型
 *    - device_type_ 的值是 DeviceType::kCPU 或 DeviceType::kCUDA 中的一种
 *    - 每次测试运行时，device_type_ 只取其中一个值
 *    - GoogleTest 会分别运行这两种场景的测试用例：
 *      * 第一次运行：device_type_ = DeviceType::kCPU，测试在 CPU 上运行
 *      * 第二次运行：device_type_ = DeviceType::kCUDA，测试在 CUDA 上运行
 *
 * 4. 测试代码使用：
 *    - 测试代码中通过deviceType()获取当前设备类型
 *    - 使用origindl API显式指定设备创建张量：
 *      Tensor(data, shape, dtype(DataType::kFloat32).device(deviceType()))
 *    - 这样，同一个测试代码会自动在两种设备上运行，确保功能在 CPU 和 CUDA 上都正确
 */
class OperatorTestBase : public ::testing::TestWithParam<DeviceType>
{
protected:
    void SetUp() override
    {
        // 从GoogleTest参数化测试获取设备类型
        // GetParam()返回当前测试实例的参数值（设备类型）
        // device_type_ 的值是 DeviceType::kCPU 或 DeviceType::kCUDA 中的一种
        // 每次测试运行时，device_type_ 只取其中一个值
        // GoogleTest会为每个设备类型生成一个测试实例，分别运行这两种场景
        device_type_ = GetParam();

        // 如果是CUDA但不可用，跳过测试
        if (device_type_ == DeviceType::kCUDA)
        {
            if (!TestUtils::isCudaAvailable())
            {
                GTEST_SKIP() << "CUDA is not available on this system";
            }
        }
    }

    void TearDown() override
    {
        if (device_type_ == DeviceType::kCUDA && TestUtils::isCudaAvailable())
        {
            cuda::synchronize();
        }
    }

    /**
     * @brief 获取当前测试的设备类型
     * @return 当前设备类型（来自GetParam()）
     * @note device_type_ 是 DeviceType 枚举类型，值可能是 DeviceType::kCPU 或 DeviceType::kCUDA 中的一种
     *       每次测试运行时，device_type_ 只取其中一个值
     *       使用方式：Tensor(data, shape, dtype(DataType::kFloat32).device(deviceType()))
     */
    DeviceType deviceType() const { return device_type_; }

    /**
     * @brief 精度容忍度
     */
    static constexpr double kTolerance = TestTolerance::kDefault;

private:
    DeviceType device_type_;  // 当前测试的设备类型（来自GetParam()）
                              // GetParam()从GoogleTest参数化测试获取设备类型
                              // 值可能是 DeviceType::kCPU 或 DeviceType::kCUDA 中的一种
                              // 每次测试运行时，device_type_ 只取其中一个值
                              // GoogleTest会为每个设备类型生成一个测试实例，分别运行这两种场景
};

/**
 * @brief 获取可用的设备列表
 * @return 可用设备的DeviceType向量（总是包含CPU，如果CUDA可用则包含CUDA）
 * @note 这个函数被INSTANTIATE_DEVICE_TEST_SUITE_P使用，生成测试参数
 */
inline std::vector<DeviceType> GetAvailableDevices()
{
    std::vector<DeviceType> devices = {DeviceType::kCPU};

    if (TestUtils::isCudaAvailable())
    {
        devices.push_back(DeviceType::kCUDA);
    }

    return devices;
}

/**
 * @brief 设备类型名称转换函数（用于测试参数显示）
 * @param device_type 设备类型
 * @return 设备类型名称字符串
 */
inline std::string DeviceTypeName(DeviceType device_type)
{
    return (device_type == DeviceType::kCPU) ? "CPU" : "CUDA";
}

/**
 * @brief 测试参数生成器宏
 * @details 自动为CPU和可用CUDA生成测试用例
 *          使用方式：INSTANTIATE_DEVICE_TEST_SUITE_P(TestSuiteName)
 *
 * 工作原理：
 *
 * 1. 参数生成阶段：
 *    - 调用GetAvailableDevices()获取可用设备列表（如{DeviceType::kCPU, DeviceType::kCUDA}）
 *    - 使用ValuesIn()为每个设备类型生成一个测试参数
 *
 * 2. 测试实例生成：
 *    - 对每个TEST_P测试，GoogleTest会为每个参数值创建一个测试实例
 *    - 例如：TEST_P(AddOperatorTest, ForwardBasic) 会生成两个测试实例：
 *      * AllDevices/AddOperatorTest.ForwardBasic/CPU
 *      * AllDevices/AddOperatorTest.ForwardBasic/CUDA（如果CUDA可用）
 *
 * 3. 测试执行阶段：
 *    - 在SetUp()中，GetParam()返回当前测试实例的参数值（设备类型）
 *    - device_type_ = GetParam() 从GoogleTest参数化测试获取设备类型
 *    - device_type_ 的值是 DeviceType::kCPU 或 DeviceType::kCUDA 中的一种
 *    - 每次测试运行时，device_type_ 只取其中一个值
 *    - GoogleTest 会分别运行这两种场景的测试用例：
 *      * 第一次运行：device_type_ = DeviceType::kCPU，测试在 CPU 上运行
 *      * 第二次运行：device_type_ = DeviceType::kCUDA，测试在 CUDA 上运行
 *
 * 4. 测试代码使用：
 *    - 测试代码中通过deviceType()获取当前设备类型
 *    - 使用origindl API显式指定设备创建张量：
 *      Tensor(data, shape, dtype(DataType::kFloat32).device(deviceType()))
 *    - 这样，同一个测试代码会自动在两种设备上运行，确保功能在 CPU 和 CUDA 上都正确
 */
#define INSTANTIATE_DEVICE_TEST_SUITE_P(TestSuiteName)                                                            \
    INSTANTIATE_TEST_SUITE_P(AllDevices, TestSuiteName, ::testing::ValuesIn(origin::test::GetAvailableDevices()), \
                             [](const ::testing::TestParamInfo<origin::DeviceType> &info) {                       \
                                 return origin::test::DeviceTypeName(info.param);                                 \
                             })

}  // namespace test
}  // namespace origin

#endif  // __ORIGIN_DL_DEVICE_TEST_BASE_H__
