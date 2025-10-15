#ifndef __ORIGIN_UTILS_STATIC_ASSERT_H__
#define __ORIGIN_UTILS_STATIC_ASSERT_H__

#include <type_traits>

namespace origin
{
namespace utils
{

/**
 * @brief 模板类型静态检查宏
 * @details 用于检查模板参数T是否为算术类型且不是指针类型
 */
#define ORIGIN_STATIC_ASSERT_ARITHMETIC(T)                                                             \
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type (int, float, double, etc.)"); \
    static_assert(!std::is_pointer_v<T>, "T cannot be a pointer type")

/**
 * @brief 检查类型是否为算术类型的宏
 * @details 仅检查是否为算术类型，不检查指针
 */
#define ORIGIN_STATIC_ASSERT_IS_ARITHMETIC(T) \
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type (int, float, double, etc.)")

/**
 * @brief 检查类型不是指针的宏
 * @details 仅检查不是指针类型
 */
#define ORIGIN_STATIC_ASSERT_NOT_POINTER(T) static_assert(!std::is_pointer_v<T>, "T cannot be a pointer type")

}  // namespace utils
}  // namespace origin

#endif  // __ORIGIN_UTILS_STATIC_ASSERT_H__
