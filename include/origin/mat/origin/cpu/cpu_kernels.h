#ifndef __ORIGIN_DL_CPU_KERNELS_H__
#define __ORIGIN_DL_CPU_KERNELS_H__

#include <cstddef>
#include "origin/mat/basic_types.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU计算函数实现
 * @details 提供CPU端的元素级计算函数，与CUDA内核功能对应但使用CPU实现
 */

/**
 * @brief CPU元素级二元运算内核实现
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入数据A
 * @param b 输入数据B
 * @param c 输出数据
 * @param n 元素数量
 * @param op 操作函数对象
 */
template <typename T, typename Op>
void cpu_elementwise_kernel(const T *a, const T *b, T *c, size_t n, Op op)
{
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = op(a[i], b[i]);
    }
}

/**
 * @brief CPU简单广播二元运算内核实现
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
void cpu_simple_broadcast_kernel(const T *a,
                                 const T *b,
                                 T *c,
                                 size_t a_elements,
                                 size_t b_elements,
                                 size_t result_elements,
                                 Op op)
{
    // 如果A是标量，B是矩阵
    if (a_elements == 1)
    {
        T scalar_a = a[0];
        for (size_t i = 0; i < result_elements; ++i)
        {
            c[i] = op(scalar_a, b[i]);
        }
    }
    // 如果B是标量，A是矩阵
    else if (b_elements == 1)
    {
        T scalar_b = b[0];
        for (size_t i = 0; i < result_elements; ++i)
        {
            c[i] = op(a[i], scalar_b);
        }
    }
    // 如果A和B都是标量
    else
    {
        c[0] = op(a[0], b[0]);
    }
}

/**
 * @brief CPU一元运算内核实现
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
 * @param a 输入数据
 * @param c 输出数据
 * @param n 元素数量
 * @param op 操作函数对象
 */
template <typename T, typename Op>
void cpu_unary_kernel(const T *a, T *c, size_t n, Op op)
{
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = op(a[i]);
    }
}

}  // namespace cpu
}  // namespace origin

#endif  // __ORIGIN_DL_CPU_KERNELS_H__
