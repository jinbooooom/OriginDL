#ifndef __ORIGIN_DL_DEVICE_COMMON_TYPE_DISPATCHER_H__
#define __ORIGIN_DL_DEVICE_COMMON_TYPE_DISPATCHER_H__

#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace device_common
{

/**
 * @brief 设备通用类型分发器，用于替代重复的switch语句
 * @details 提供统一的类型分发机制，支持编译时类型安全，CPU、GPU等所有设备通用
 */
class TypeDispatcher
{
public:
    /**
     * @brief 根据数据类型分发到对应的模板函数
     * @tparam Func 函数对象类型，必须实现template<typename T> auto operator()()方法
     * @param dtype 数据类型
     * @param func 函数对象
     * @return 函数执行结果
     */
    template <typename Func>
    static auto dispatch(DataType dtype, Func &&func)
    {
        switch (dtype)
        {
            case DataType::kFloat32:
                return func.template operator()<float>();
            case DataType::kFloat64:
                return func.template operator()<double>();
            case DataType::kInt32:
                return func.template operator()<int32_t>();
            case DataType::kInt8:
                return func.template operator()<int8_t>();
            case DataType::kInt64:
                return func.template operator()<int64_t>();
            case DataType::kUInt8:
                return func.template operator()<uint8_t>();
            default:
                THROW_INVALID_ARG("Unsupported data type {} for operation", dtype_to_string(dtype));
        }
    }

    /**
     * @brief 根据数据类型分发到对应的模板函数（无返回值版本）
     * @tparam Func 函数对象类型，必须实现template<typename T> void operator()()方法
     * @param dtype 数据类型
     * @param func 函数对象
     */
    template <typename Func>
    static void dispatch_void(DataType dtype, Func &&func)
    {
        switch (dtype)
        {
            case DataType::kFloat32:
                /*
                由于 func 是一个函数对象，不是直接的函数指针，所以需要显式调用 operator() 方法。

                关于.template，当编译器遇到func.operator()<float>()时，编译器不知道 < 和 > 的含义，它可能被理解为：
                选项A：模板参数
                选项B：比较操作符
                为了避免歧义，需要使用 .template 来显式告诉编译器这是一个模板调用，按照模板语义解析而不是比较操作符。
                */
                func.template operator()<float>();  // 理解为 func<float>()
                break;
            case DataType::kFloat64:
                func.template operator()<double>();  // 理解为 func<double>()
                break;
            case DataType::kInt32:
                func.template operator()<int32_t>();  // 理解为 func<int32_t>()
                break;
            case DataType::kInt8:
                func.template operator()<int8_t>();  // 理解为 func<int8_t>()
                break;
            case DataType::kInt64:
                func.template operator()<int64_t>();
                break;
            case DataType::kUInt8:
                func.template operator()<uint8_t>();
                break;
            default:
                THROW_INVALID_ARG("Unsupported data type {} for operation", dtype_to_string(dtype));
        }
    }
};

}  // namespace device_common
}  // namespace origin

#endif  // __ORIGIN_DL_DEVICE_COMMON_TYPE_DISPATCHER_H__
