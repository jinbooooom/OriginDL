#include "origin/mat/origin/cpu/cpu_kernels.h"
#include "origin/mat/origin/cpu/operation_templates.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU元素级二元运算内核实现
 * @tparam T 数据类型
 * @tparam Op 操作函数对象类型
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
 */
template <typename T, typename Op>
void cpu_simple_broadcast_kernel(const T *a, const T *b, T *c, 
                                size_t a_elements, size_t b_elements, 
                                size_t result_elements, Op op)
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
 */
template <typename T, typename Op>
void cpu_unary_kernel(const T *a, T *c, size_t n, Op op)
{
    for (size_t i = 0; i < n; ++i)
    {
        c[i] = op(a[i]);
    }
}

// 显式实例化 - 按照 DataType 枚举顺序排列
// AddOp 实例化
template void cpu_elementwise_kernel<float, AddOp>(const float *, const float *, float *, size_t, AddOp);
template void cpu_elementwise_kernel<double, AddOp>(const double *, const double *, double *, size_t, AddOp);
template void cpu_elementwise_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, AddOp);
template void cpu_elementwise_kernel<int16_t, AddOp>(const int16_t *, const int16_t *, int16_t *, size_t, AddOp);
template void cpu_elementwise_kernel<int32_t, AddOp>(const int32_t *, const int32_t *, int32_t *, size_t, AddOp);
template void cpu_elementwise_kernel<int64_t, AddOp>(const int64_t *, const int64_t *, int64_t *, size_t, AddOp);
template void cpu_elementwise_kernel<uint8_t, AddOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, AddOp);
template void cpu_elementwise_kernel<uint16_t, AddOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, AddOp);
template void cpu_elementwise_kernel<uint32_t, AddOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, AddOp);
template void cpu_elementwise_kernel<uint64_t, AddOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, AddOp);
template void cpu_elementwise_kernel<bool, AddOp>(const bool *, const bool *, bool *, size_t, AddOp);

// SubtractOp 实例化
template void cpu_elementwise_kernel<float, SubtractOp>(const float *, const float *, float *, size_t, SubtractOp);
template void cpu_elementwise_kernel<double, SubtractOp>(const double *, const double *, double *, size_t, SubtractOp);
template void cpu_elementwise_kernel<int8_t, SubtractOp>(const int8_t *, const int8_t *, int8_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<int16_t, SubtractOp>(const int16_t *, const int16_t *, int16_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<int32_t, SubtractOp>(const int32_t *, const int32_t *, int32_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<int64_t, SubtractOp>(const int64_t *, const int64_t *, int64_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<uint8_t, SubtractOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<uint16_t, SubtractOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<uint32_t, SubtractOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<uint64_t, SubtractOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, SubtractOp);
template void cpu_elementwise_kernel<bool, SubtractOp>(const bool *, const bool *, bool *, size_t, SubtractOp);

// MultiplyOp 实例化
template void cpu_elementwise_kernel<float, MultiplyOp>(const float *, const float *, float *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<double, MultiplyOp>(const double *, const double *, double *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<int8_t, MultiplyOp>(const int8_t *, const int8_t *, int8_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<int16_t, MultiplyOp>(const int16_t *, const int16_t *, int16_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<int32_t, MultiplyOp>(const int32_t *, const int32_t *, int32_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<int64_t, MultiplyOp>(const int64_t *, const int64_t *, int64_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<uint8_t, MultiplyOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<uint16_t, MultiplyOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<uint32_t, MultiplyOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<uint64_t, MultiplyOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, MultiplyOp);
template void cpu_elementwise_kernel<bool, MultiplyOp>(const bool *, const bool *, bool *, size_t, MultiplyOp);

// DivideOp 实例化
template void cpu_elementwise_kernel<float, DivideOp>(const float *, const float *, float *, size_t, DivideOp);
template void cpu_elementwise_kernel<double, DivideOp>(const double *, const double *, double *, size_t, DivideOp);
template void cpu_elementwise_kernel<int8_t, DivideOp>(const int8_t *, const int8_t *, int8_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<int16_t, DivideOp>(const int16_t *, const int16_t *, int16_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<int32_t, DivideOp>(const int32_t *, const int32_t *, int32_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<int64_t, DivideOp>(const int64_t *, const int64_t *, int64_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<uint8_t, DivideOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<uint16_t, DivideOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<uint32_t, DivideOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<uint64_t, DivideOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, DivideOp);
template void cpu_elementwise_kernel<bool, DivideOp>(const bool *, const bool *, bool *, size_t, DivideOp);

// 简单广播内核实例化
template void cpu_simple_broadcast_kernel<float, AddOp>(const float *, const float *, float *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<double, AddOp>(const double *, const double *, double *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<int16_t, AddOp>(const int16_t *, const int16_t *, int16_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<int32_t, AddOp>(const int32_t *, const int32_t *, int32_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<int64_t, AddOp>(const int64_t *, const int64_t *, int64_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<uint8_t, AddOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<uint16_t, AddOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<uint32_t, AddOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<uint64_t, AddOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, size_t, size_t, AddOp);
template void cpu_simple_broadcast_kernel<bool, AddOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, AddOp);

template void cpu_simple_broadcast_kernel<float, SubtractOp>(const float *, const float *, float *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<double, SubtractOp>(const double *, const double *, double *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<int8_t, SubtractOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<int16_t, SubtractOp>(const int16_t *, const int16_t *, int16_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<int32_t, SubtractOp>(const int32_t *, const int32_t *, int32_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<int64_t, SubtractOp>(const int64_t *, const int64_t *, int64_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<uint8_t, SubtractOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<uint16_t, SubtractOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<uint32_t, SubtractOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<uint64_t, SubtractOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, size_t, size_t, SubtractOp);
template void cpu_simple_broadcast_kernel<bool, SubtractOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, SubtractOp);

template void cpu_simple_broadcast_kernel<float, MultiplyOp>(const float *, const float *, float *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<double, MultiplyOp>(const double *, const double *, double *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<int8_t, MultiplyOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<int16_t, MultiplyOp>(const int16_t *, const int16_t *, int16_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<int32_t, MultiplyOp>(const int32_t *, const int32_t *, int32_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<int64_t, MultiplyOp>(const int64_t *, const int64_t *, int64_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<uint8_t, MultiplyOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<uint16_t, MultiplyOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<uint32_t, MultiplyOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<uint64_t, MultiplyOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, size_t, size_t, MultiplyOp);
template void cpu_simple_broadcast_kernel<bool, MultiplyOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, MultiplyOp);

template void cpu_simple_broadcast_kernel<float, DivideOp>(const float *, const float *, float *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<double, DivideOp>(const double *, const double *, double *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<int8_t, DivideOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<int16_t, DivideOp>(const int16_t *, const int16_t *, int16_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<int32_t, DivideOp>(const int32_t *, const int32_t *, int32_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<int64_t, DivideOp>(const int64_t *, const int64_t *, int64_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<uint8_t, DivideOp>(const uint8_t *, const uint8_t *, uint8_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<uint16_t, DivideOp>(const uint16_t *, const uint16_t *, uint16_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<uint32_t, DivideOp>(const uint32_t *, const uint32_t *, uint32_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<uint64_t, DivideOp>(const uint64_t *, const uint64_t *, uint64_t *, size_t, size_t, size_t, DivideOp);
template void cpu_simple_broadcast_kernel<bool, DivideOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, DivideOp);

// 一元内核实例化
template void cpu_unary_kernel<float, NegOp>(const float *, float *, size_t, NegOp);
template void cpu_unary_kernel<double, NegOp>(const double *, double *, size_t, NegOp);
template void cpu_unary_kernel<int8_t, NegOp>(const int8_t *, int8_t *, size_t, NegOp);
template void cpu_unary_kernel<int16_t, NegOp>(const int16_t *, int16_t *, size_t, NegOp);
template void cpu_unary_kernel<int32_t, NegOp>(const int32_t *, int32_t *, size_t, NegOp);
template void cpu_unary_kernel<int64_t, NegOp>(const int64_t *, int64_t *, size_t, NegOp);
template void cpu_unary_kernel<uint8_t, NegOp>(const uint8_t *, uint8_t *, size_t, NegOp);
template void cpu_unary_kernel<uint16_t, NegOp>(const uint16_t *, uint16_t *, size_t, NegOp);
template void cpu_unary_kernel<uint32_t, NegOp>(const uint32_t *, uint32_t *, size_t, NegOp);
template void cpu_unary_kernel<uint64_t, NegOp>(const uint64_t *, uint64_t *, size_t, NegOp);
template void cpu_unary_kernel<bool, NegOp>(const bool *, bool *, size_t, NegOp);

template void cpu_unary_kernel<float, SquareOp>(const float *, float *, size_t, SquareOp);
template void cpu_unary_kernel<double, SquareOp>(const double *, double *, size_t, SquareOp);
template void cpu_unary_kernel<int8_t, SquareOp>(const int8_t *, int8_t *, size_t, SquareOp);
template void cpu_unary_kernel<int16_t, SquareOp>(const int16_t *, int16_t *, size_t, SquareOp);
template void cpu_unary_kernel<int32_t, SquareOp>(const int32_t *, int32_t *, size_t, SquareOp);
template void cpu_unary_kernel<int64_t, SquareOp>(const int64_t *, int64_t *, size_t, SquareOp);
template void cpu_unary_kernel<uint8_t, SquareOp>(const uint8_t *, uint8_t *, size_t, SquareOp);
template void cpu_unary_kernel<uint16_t, SquareOp>(const uint16_t *, uint16_t *, size_t, SquareOp);
template void cpu_unary_kernel<uint32_t, SquareOp>(const uint32_t *, uint32_t *, size_t, SquareOp);
template void cpu_unary_kernel<uint64_t, SquareOp>(const uint64_t *, uint64_t *, size_t, SquareOp);
template void cpu_unary_kernel<bool, SquareOp>(const bool *, bool *, size_t, SquareOp);

// ExpOp
template void cpu_unary_kernel<float, ExpOp>(const float *, float *, size_t, ExpOp);
template void cpu_unary_kernel<double, ExpOp>(const double *, double *, size_t, ExpOp);
template void cpu_unary_kernel<int8_t, ExpOp>(const int8_t *, int8_t *, size_t, ExpOp);
template void cpu_unary_kernel<int16_t, ExpOp>(const int16_t *, int16_t *, size_t, ExpOp);
template void cpu_unary_kernel<int32_t, ExpOp>(const int32_t *, int32_t *, size_t, ExpOp);
template void cpu_unary_kernel<int64_t, ExpOp>(const int64_t *, int64_t *, size_t, ExpOp);
template void cpu_unary_kernel<uint8_t, ExpOp>(const uint8_t *, uint8_t *, size_t, ExpOp);
template void cpu_unary_kernel<uint16_t, ExpOp>(const uint16_t *, uint16_t *, size_t, ExpOp);
template void cpu_unary_kernel<uint32_t, ExpOp>(const uint32_t *, uint32_t *, size_t, ExpOp);
template void cpu_unary_kernel<uint64_t, ExpOp>(const uint64_t *, uint64_t *, size_t, ExpOp);
template void cpu_unary_kernel<bool, ExpOp>(const bool *, bool *, size_t, ExpOp);

// LogOp
template void cpu_unary_kernel<float, LogOp>(const float *, float *, size_t, LogOp);
template void cpu_unary_kernel<double, LogOp>(const double *, double *, size_t, LogOp);
template void cpu_unary_kernel<int8_t, LogOp>(const int8_t *, int8_t *, size_t, LogOp);
template void cpu_unary_kernel<int16_t, LogOp>(const int16_t *, int16_t *, size_t, LogOp);
template void cpu_unary_kernel<int32_t, LogOp>(const int32_t *, int32_t *, size_t, LogOp);
template void cpu_unary_kernel<int64_t, LogOp>(const int64_t *, int64_t *, size_t, LogOp);
template void cpu_unary_kernel<uint8_t, LogOp>(const uint8_t *, uint8_t *, size_t, LogOp);
template void cpu_unary_kernel<uint16_t, LogOp>(const uint16_t *, uint16_t *, size_t, LogOp);
template void cpu_unary_kernel<uint32_t, LogOp>(const uint32_t *, uint32_t *, size_t, LogOp);
template void cpu_unary_kernel<uint64_t, LogOp>(const uint64_t *, uint64_t *, size_t, LogOp);
template void cpu_unary_kernel<bool, LogOp>(const bool *, bool *, size_t, LogOp);

// SqrtOp
template void cpu_unary_kernel<float, SqrtOp>(const float *, float *, size_t, SqrtOp);
template void cpu_unary_kernel<double, SqrtOp>(const double *, double *, size_t, SqrtOp);
template void cpu_unary_kernel<int8_t, SqrtOp>(const int8_t *, int8_t *, size_t, SqrtOp);
template void cpu_unary_kernel<int16_t, SqrtOp>(const int16_t *, int16_t *, size_t, SqrtOp);
template void cpu_unary_kernel<int32_t, SqrtOp>(const int32_t *, int32_t *, size_t, SqrtOp);
template void cpu_unary_kernel<int64_t, SqrtOp>(const int64_t *, int64_t *, size_t, SqrtOp);
template void cpu_unary_kernel<uint8_t, SqrtOp>(const uint8_t *, uint8_t *, size_t, SqrtOp);
template void cpu_unary_kernel<uint16_t, SqrtOp>(const uint16_t *, uint16_t *, size_t, SqrtOp);
template void cpu_unary_kernel<uint32_t, SqrtOp>(const uint32_t *, uint32_t *, size_t, SqrtOp);
template void cpu_unary_kernel<uint64_t, SqrtOp>(const uint64_t *, uint64_t *, size_t, SqrtOp);
template void cpu_unary_kernel<bool, SqrtOp>(const bool *, bool *, size_t, SqrtOp);

}  // namespace cpu
}  // namespace origin
