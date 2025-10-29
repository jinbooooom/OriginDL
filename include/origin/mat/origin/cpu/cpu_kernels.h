#ifndef __ORIGIN_DL_CPU_KERNELS_H__
#define __ORIGIN_DL_CPU_KERNELS_H__

#include <cstddef>
#include "origin/mat/basic_types.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU内核函数声明
 * @details 提供与CUDA内核对应的CPU实现，保持接口一致性
 */

/**
 * @brief CPU元素级二元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入数据A
 * @param b 输入数据B
 * @param c 输出数据
 * @param n 元素数量
 * @param op 操作函数对象
 */
template <typename T, typename Op>
void cpu_elementwise_kernel(const T *a, const T *b, T *c, size_t n, Op op);

/**
 * @brief CPU简单广播二元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入数据A
 * @param b 输入数据B
 * @param c 输出数据
 * @param a_elements A的元素数量
 * @param b_elements B的元素数量
 * @param result_elements 结果元素数量
 * @param op 操作函数对象
 */
template <typename T, typename Op>
void cpu_simple_broadcast_kernel(const T *a, const T *b, T *c, 
                                size_t a_elements, size_t b_elements, 
                                size_t result_elements, Op op);

/**
 * @brief CPU一元运算内核
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入数据
 * @param c 输出数据
 * @param n 元素数量
 * @param op 操作函数对象
 */
template <typename T, typename Op>
void cpu_unary_kernel(const T *a, T *c, size_t n, Op op);

}  // namespace cpu
}  // namespace origin

#endif  // __ORIGIN_DL_CPU_KERNELS_H__
