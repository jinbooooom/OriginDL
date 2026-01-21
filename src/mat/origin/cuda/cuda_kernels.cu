#include <cmath>
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

// ============================================================================
// 基础元素级运算内核实现
// ============================================================================

/**
 * @brief 基础元素级二元运算内核实现
 * @details 使用合并内存访问模式，确保相邻线程访问相邻内存地址
 */
template <typename T, typename Op>
__global__ void elementwise_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, size_t n, Op op)
{
    // 计算全局线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 边界检查，确保不越界
    if (idx < n)
    {
        // 执行元素级操作
        // 使用__restrict__确保编译器知道指针不会重叠，允许更激进的优化
        c[idx] = op(a[idx], b[idx]);
    }
}

/**
 * @brief 基础元素级一元运算内核实现
 */
template <typename T, typename Op>
__global__ void unary_kernel(const T *__restrict__ a, T *__restrict__ c, size_t n, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = op(a[idx]);
    }
}

/**
 * @brief 类型转换内核实现
 * @details 在GPU上直接进行类型转换，避免CPU-CUDA数据传输
 */
template <typename SrcT, typename DstT>
__global__ void type_conversion_kernel(const SrcT *__restrict__ src, DstT *__restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        dst[idx] = static_cast<DstT>(src[idx]);
    }
}

/**
 * @brief 索引写入内核实现（单个元素）
 */
template <typename T>
__global__ void index_put_kernel(T *data, size_t index, T value)
{
    // 只写入指定索引位置的值
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        data[index] = value;
    }
}

/**
 * @brief 启动索引写入内核实现（单个元素）
 */
template <typename T>
void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream)
{
    // 使用1个block，1个thread来写入单个元素
    index_put_kernel<T><<<1, 1, 0, stream>>>(data, index, value);
}

/**
 * @brief 标量运算内核实现
 */
template <typename T, typename Op>
__global__ void scalar_kernel(const T *__restrict__ a, T scalar, T *__restrict__ c, size_t n, Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = op(a[idx], scalar);
    }
}

// ============================================================================
// 广播运算内核实现
// ============================================================================

/**
 * @brief 简单广播内核实现（一个操作数是标量）
 * @details 用于处理简单广播情况，其中一个操作数是标量（只有1个元素）。
 *          这是广播运算中最常见的情况，性能最优。
 * @note 此内核假设其中一个张量是标量，另一个张量是普通张量。
 *       如果两个张量都是标量，此内核仍然可以正常工作。
 */
template <typename T, typename Op>
__global__ void simple_broadcast_kernel(const T *__restrict__ a,
                                        const T *__restrict__ b,
                                        T *__restrict__ c,
                                        size_t a_elements,
                                        size_t b_elements,
                                        size_t c_elements,
                                        Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < c_elements)
    {
        // 计算源索引：如果操作数是标量，索引为0，否则为当前索引
        size_t a_idx = (a_elements == 1) ? 0 : idx;
        size_t b_idx = (b_elements == 1) ? 0 : idx;

        c[idx] = op(a[a_idx], b[b_idx]);
    }
}

/**
 * @brief 复杂广播内核实现（处理不同维度的张量）
 */
template <typename T, typename Op>
__global__ void complex_broadcast_kernel(const T *__restrict__ a,
                                         const T *__restrict__ b,
                                         T *__restrict__ c,
                                         const int *a_strides,
                                         const int *b_strides,
                                         const int *c_strides,
                                         const int *a_shape,
                                         const int *b_shape,
                                         const int *c_shape,
                                         int ndims,
                                         size_t total_elements,
                                         Op op)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 将线性索引转换为多维索引
        int c_indices[8];  // 假设最大8维
        int temp = idx;
        for (int i = ndims - 1; i >= 0; --i)
        {
            c_indices[i] = temp % c_shape[i];
            temp /= c_shape[i];
        }

        // 计算a和b的线性索引
        int a_idx = 0, b_idx = 0;
        for (int i = 0; i < ndims; ++i)
        {
            // 如果维度大小为1，则索引为0（广播）
            int a_dim_idx = (a_shape[i] == 1) ? 0 : c_indices[i];
            int b_dim_idx = (b_shape[i] == 1) ? 0 : c_indices[i];

            a_idx += a_dim_idx * a_strides[i];
            b_idx += b_dim_idx * b_strides[i];
        }

        c[idx] = op(a[a_idx], b[b_idx]);
    }
}

// ============================================================================
// 内核启动函数实现
// ============================================================================

/**
 * @brief 启动元素级二元运算内核
 */
template <typename T, typename Op>
void launch_elementwise_kernel(const T *a, const T *b, T *c, size_t n, Op op, cudaStream_t stream)
{
    // 根据数据大小选择最优的线程块大小
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    // 启动内核
    elementwise_kernel<T, Op><<<grid, block, 0, stream>>>(a, b, c, n, op);
}

/**
 * @brief 启动元素级一元运算内核
 */
template <typename T, typename Op>
void launch_unary_kernel(const T *a, T *c, size_t n, Op op, cudaStream_t stream)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    unary_kernel<T, Op><<<grid, block, 0, stream>>>(a, c, n, op);
}

/**
 * @brief 启动类型转换内核
 */
template <typename SrcT, typename DstT>
void launch_type_conversion_kernel(const SrcT *src, DstT *dst, size_t n, cudaStream_t stream)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    type_conversion_kernel<SrcT, DstT><<<grid, block, 0, stream>>>(src, dst, n);
}

/**
 * @brief 启动标量运算内核
 */
template <typename T, typename Op>
void launch_scalar_kernel(const T *a, T scalar, T *c, size_t n, Op op, cudaStream_t stream)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    scalar_kernel<T, Op><<<grid, block, 0, stream>>>(a, scalar, c, n, op);
}

/**
 * @brief 启动简单广播内核
 * @details 用于启动简单广播运算，其中一个操作数是标量。
 *          自动计算最优的网格和块大小。
 */
template <typename T, typename Op>
void launch_simple_broadcast_kernel(const T *a,
                                    const T *b,
                                    T *c,
                                    size_t a_elements,
                                    size_t b_elements,
                                    size_t c_elements,
                                    Op op,
                                    cudaStream_t stream)
{
    dim3 block = get_optimal_block_size(c_elements);
    dim3 grid  = get_optimal_grid_size(c_elements, block);

    simple_broadcast_kernel<T, Op><<<grid, block, 0, stream>>>(a, b, c, a_elements, b_elements, c_elements, op);
}

/**
 * @brief 启动复杂广播内核
 * @details 用于启动复杂广播运算，处理不同维度的张量之间的广播。
 *          需要预先计算步长和形状信息，适用于任意维度的广播。
 * @note 此函数需要调用者确保步长和形状数组在设备内存中，且维度信息正确。
 *       建议使用 compute_broadcast_strides 等辅助函数来计算这些参数。
 */
template <typename T, typename Op>
void launch_complex_broadcast_kernel(const T *a,
                                     const T *b,
                                     T *c,
                                     const int *a_strides,
                                     const int *b_strides,
                                     const int *c_strides,
                                     const int *a_shape,
                                     const int *b_shape,
                                     const int *c_shape,
                                     int ndims,
                                     size_t total_elements,
                                     Op op,
                                     cudaStream_t stream)
{
    dim3 block = get_optimal_block_size(total_elements);
    dim3 grid  = get_optimal_grid_size(total_elements, block);

    complex_broadcast_kernel<T, Op><<<grid, block, 0, stream>>>(a, b, c, a_strides, b_strides, c_strides, a_shape,
                                                                b_shape, c_shape, ndims, total_elements, op);
}

// ============================================================================
// 显式模板实例化
// ============================================================================

// 基础内核的显式实例化
// 按照 DataType 枚举顺序排列
template __global__ void elementwise_kernel<float, AddOp>(const float *,
                                                          const float *,
                                                          float *,
                                                          size_t,
                                                          AddOp);  // kFloat32
template __global__ void elementwise_kernel<double, AddOp>(const double *,
                                                           const double *,
                                                           double *,
                                                           size_t,
                                                           AddOp);  // kFloat64/kDouble
template __global__ void elementwise_kernel<int8_t, AddOp>(const int8_t *,
                                                           const int8_t *,
                                                           int8_t *,
                                                           size_t,
                                                           AddOp);  // kInt8
template __global__ void elementwise_kernel<int16_t, AddOp>(const int16_t *,
                                                            const int16_t *,
                                                            int16_t *,
                                                            size_t,
                                                            AddOp);  // kInt16
template __global__ void elementwise_kernel<int32_t, AddOp>(const int32_t *,
                                                            const int32_t *,
                                                            int32_t *,
                                                            size_t,
                                                            AddOp);  // kInt32
template __global__ void elementwise_kernel<int64_t, AddOp>(const int64_t *,
                                                            const int64_t *,
                                                            int64_t *,
                                                            size_t,
                                                            AddOp);  // kInt64
template __global__ void elementwise_kernel<uint8_t, AddOp>(const uint8_t *,
                                                            const uint8_t *,
                                                            uint8_t *,
                                                            size_t,
                                                            AddOp);  // kUInt8
template __global__ void elementwise_kernel<uint16_t, AddOp>(const uint16_t *,
                                                             const uint16_t *,
                                                             uint16_t *,
                                                             size_t,
                                                             AddOp);  // kUInt16
template __global__ void elementwise_kernel<uint32_t, AddOp>(const uint32_t *,
                                                             const uint32_t *,
                                                             uint32_t *,
                                                             size_t,
                                                             AddOp);  // kUInt32
template __global__ void elementwise_kernel<uint64_t, AddOp>(const uint64_t *,
                                                             const uint64_t *,
                                                             uint64_t *,
                                                             size_t,
                                                             AddOp);                                          // kUInt64
template __global__ void elementwise_kernel<bool, AddOp>(const bool *, const bool *, bool *, size_t, AddOp);  // kBool

template __global__ void elementwise_kernel<float, SubtractOp>(const float *,
                                                               const float *,
                                                               float *,
                                                               size_t,
                                                               SubtractOp);  // kFloat32
template __global__ void elementwise_kernel<double, SubtractOp>(const double *,
                                                                const double *,
                                                                double *,
                                                                size_t,
                                                                SubtractOp);  // kFloat64/kDouble
template __global__ void elementwise_kernel<int8_t, SubtractOp>(const int8_t *,
                                                                const int8_t *,
                                                                int8_t *,
                                                                size_t,
                                                                SubtractOp);  // kInt8
template __global__ void elementwise_kernel<int16_t, SubtractOp>(const int16_t *,
                                                                 const int16_t *,
                                                                 int16_t *,
                                                                 size_t,
                                                                 SubtractOp);  // kInt16
template __global__ void elementwise_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                 const int32_t *,
                                                                 int32_t *,
                                                                 size_t,
                                                                 SubtractOp);  // kInt32
template __global__ void elementwise_kernel<int64_t, SubtractOp>(const int64_t *,
                                                                 const int64_t *,
                                                                 int64_t *,
                                                                 size_t,
                                                                 SubtractOp);  // kInt64
template __global__ void elementwise_kernel<uint8_t, SubtractOp>(const uint8_t *,
                                                                 const uint8_t *,
                                                                 uint8_t *,
                                                                 size_t,
                                                                 SubtractOp);  // kUInt8
template __global__ void elementwise_kernel<uint16_t, SubtractOp>(const uint16_t *,
                                                                  const uint16_t *,
                                                                  uint16_t *,
                                                                  size_t,
                                                                  SubtractOp);  // kUInt16
template __global__ void elementwise_kernel<uint32_t, SubtractOp>(const uint32_t *,
                                                                  const uint32_t *,
                                                                  uint32_t *,
                                                                  size_t,
                                                                  SubtractOp);  // kUInt32
template __global__ void elementwise_kernel<uint64_t, SubtractOp>(const uint64_t *,
                                                                  const uint64_t *,
                                                                  uint64_t *,
                                                                  size_t,
                                                                  SubtractOp);  // kUInt64
template __global__ void elementwise_kernel<bool, SubtractOp>(const bool *,
                                                              const bool *,
                                                              bool *,
                                                              size_t,
                                                              SubtractOp);  // kBool

template __global__ void elementwise_kernel<float, MultiplyOp>(const float *,
                                                               const float *,
                                                               float *,
                                                               size_t,
                                                               MultiplyOp);  // kFloat32
template __global__ void elementwise_kernel<double, MultiplyOp>(const double *,
                                                                const double *,
                                                                double *,
                                                                size_t,
                                                                MultiplyOp);  // kFloat64/kDouble
template __global__ void elementwise_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                                const int8_t *,
                                                                int8_t *,
                                                                size_t,
                                                                MultiplyOp);  // kInt8
template __global__ void elementwise_kernel<int16_t, MultiplyOp>(const int16_t *,
                                                                 const int16_t *,
                                                                 int16_t *,
                                                                 size_t,
                                                                 MultiplyOp);  // kInt16
template __global__ void elementwise_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                 const int32_t *,
                                                                 int32_t *,
                                                                 size_t,
                                                                 MultiplyOp);  // kInt32
template __global__ void elementwise_kernel<int64_t, MultiplyOp>(const int64_t *,
                                                                 const int64_t *,
                                                                 int64_t *,
                                                                 size_t,
                                                                 MultiplyOp);  // kInt64
template __global__ void elementwise_kernel<uint8_t, MultiplyOp>(const uint8_t *,
                                                                 const uint8_t *,
                                                                 uint8_t *,
                                                                 size_t,
                                                                 MultiplyOp);  // kUInt8
template __global__ void elementwise_kernel<uint16_t, MultiplyOp>(const uint16_t *,
                                                                  const uint16_t *,
                                                                  uint16_t *,
                                                                  size_t,
                                                                  MultiplyOp);  // kUInt16
template __global__ void elementwise_kernel<uint32_t, MultiplyOp>(const uint32_t *,
                                                                  const uint32_t *,
                                                                  uint32_t *,
                                                                  size_t,
                                                                  MultiplyOp);  // kUInt32
template __global__ void elementwise_kernel<uint64_t, MultiplyOp>(const uint64_t *,
                                                                  const uint64_t *,
                                                                  uint64_t *,
                                                                  size_t,
                                                                  MultiplyOp);  // kUInt64
template __global__ void elementwise_kernel<bool, MultiplyOp>(const bool *,
                                                              const bool *,
                                                              bool *,
                                                              size_t,
                                                              MultiplyOp);  // kBool

template __global__ void elementwise_kernel<float, DivideOp>(const float *,
                                                             const float *,
                                                             float *,
                                                             size_t,
                                                             DivideOp);  // kFloat32
template __global__ void elementwise_kernel<double, DivideOp>(const double *,
                                                              const double *,
                                                              double *,
                                                              size_t,
                                                              DivideOp);  // kFloat64/kDouble
template __global__ void elementwise_kernel<int8_t, DivideOp>(const int8_t *,
                                                              const int8_t *,
                                                              int8_t *,
                                                              size_t,
                                                              DivideOp);  // kInt8
template __global__ void elementwise_kernel<int16_t, DivideOp>(const int16_t *,
                                                               const int16_t *,
                                                               int16_t *,
                                                               size_t,
                                                               DivideOp);  // kInt16
template __global__ void elementwise_kernel<int32_t, DivideOp>(const int32_t *,
                                                               const int32_t *,
                                                               int32_t *,
                                                               size_t,
                                                               DivideOp);  // kInt32
template __global__ void elementwise_kernel<int64_t, DivideOp>(const int64_t *,
                                                               const int64_t *,
                                                               int64_t *,
                                                               size_t,
                                                               DivideOp);  // kInt64
template __global__ void elementwise_kernel<uint8_t, DivideOp>(const uint8_t *,
                                                               const uint8_t *,
                                                               uint8_t *,
                                                               size_t,
                                                               DivideOp);  // kUInt8
template __global__ void elementwise_kernel<uint16_t, DivideOp>(const uint16_t *,
                                                                const uint16_t *,
                                                                uint16_t *,
                                                                size_t,
                                                                DivideOp);  // kUInt16
template __global__ void elementwise_kernel<uint32_t, DivideOp>(const uint32_t *,
                                                                const uint32_t *,
                                                                uint32_t *,
                                                                size_t,
                                                                DivideOp);  // kUInt32
template __global__ void elementwise_kernel<uint64_t, DivideOp>(const uint64_t *,
                                                                const uint64_t *,
                                                                uint64_t *,
                                                                size_t,
                                                                DivideOp);  // kUInt64
template __global__ void elementwise_kernel<bool, DivideOp>(const bool *,
                                                            const bool *,
                                                            bool *,
                                                            size_t,
                                                            DivideOp);  // kBool

// 一元内核的显式实例化
// 按照 DataType 枚举顺序排列
template __global__ void unary_kernel<float, ExpOp>(const float *, float *, size_t, ExpOp);        // kFloat32
template __global__ void unary_kernel<double, ExpOp>(const double *, double *, size_t, ExpOp);     // kFloat64/kDouble
template __global__ void unary_kernel<int8_t, ExpOp>(const int8_t *, int8_t *, size_t, ExpOp);     // kInt8
template __global__ void unary_kernel<int16_t, ExpOp>(const int16_t *, int16_t *, size_t, ExpOp);  // kInt16
template __global__ void unary_kernel<int32_t, ExpOp>(const int32_t *, int32_t *, size_t, ExpOp);  // kInt32
template __global__ void unary_kernel<int64_t, ExpOp>(const int64_t *, int64_t *, size_t, ExpOp);  // kInt64
template __global__ void unary_kernel<uint8_t, ExpOp>(const uint8_t *, uint8_t *, size_t, ExpOp);  // kUInt8
template __global__ void unary_kernel<uint16_t, ExpOp>(const uint16_t *, uint16_t *, size_t, ExpOp);  // kUInt16
template __global__ void unary_kernel<uint32_t, ExpOp>(const uint32_t *, uint32_t *, size_t, ExpOp);  // kUInt32
template __global__ void unary_kernel<uint64_t, ExpOp>(const uint64_t *, uint64_t *, size_t, ExpOp);  // kUInt64
template __global__ void unary_kernel<bool, ExpOp>(const bool *, bool *, size_t, ExpOp);              // kBool

template __global__ void unary_kernel<float, LogOp>(const float *, float *, size_t, LogOp);        // kFloat32
template __global__ void unary_kernel<double, LogOp>(const double *, double *, size_t, LogOp);     // kFloat64/kDouble
template __global__ void unary_kernel<int8_t, LogOp>(const int8_t *, int8_t *, size_t, LogOp);     // kInt8
template __global__ void unary_kernel<int16_t, LogOp>(const int16_t *, int16_t *, size_t, LogOp);  // kInt16
template __global__ void unary_kernel<int32_t, LogOp>(const int32_t *, int32_t *, size_t, LogOp);  // kInt32
template __global__ void unary_kernel<int64_t, LogOp>(const int64_t *, int64_t *, size_t, LogOp);  // kInt64
template __global__ void unary_kernel<uint8_t, LogOp>(const uint8_t *, uint8_t *, size_t, LogOp);  // kUInt8
template __global__ void unary_kernel<uint16_t, LogOp>(const uint16_t *, uint16_t *, size_t, LogOp);  // kUInt16
template __global__ void unary_kernel<uint32_t, LogOp>(const uint32_t *, uint32_t *, size_t, LogOp);  // kUInt32
template __global__ void unary_kernel<uint64_t, LogOp>(const uint64_t *, uint64_t *, size_t, LogOp);  // kUInt64
template __global__ void unary_kernel<bool, LogOp>(const bool *, bool *, size_t, LogOp);              // kBool

template __global__ void unary_kernel<float, SqrtOp>(const float *, float *, size_t, SqrtOp);        // kFloat32
template __global__ void unary_kernel<double, SqrtOp>(const double *, double *, size_t, SqrtOp);     // kFloat64/kDouble
template __global__ void unary_kernel<int8_t, SqrtOp>(const int8_t *, int8_t *, size_t, SqrtOp);     // kInt8
template __global__ void unary_kernel<int16_t, SqrtOp>(const int16_t *, int16_t *, size_t, SqrtOp);  // kInt16
template __global__ void unary_kernel<int32_t, SqrtOp>(const int32_t *, int32_t *, size_t, SqrtOp);  // kInt32
template __global__ void unary_kernel<int64_t, SqrtOp>(const int64_t *, int64_t *, size_t, SqrtOp);  // kInt64
template __global__ void unary_kernel<uint8_t, SqrtOp>(const uint8_t *, uint8_t *, size_t, SqrtOp);  // kUInt8
template __global__ void unary_kernel<uint16_t, SqrtOp>(const uint16_t *, uint16_t *, size_t, SqrtOp);  // kUInt16
template __global__ void unary_kernel<uint32_t, SqrtOp>(const uint32_t *, uint32_t *, size_t, SqrtOp);  // kUInt32
template __global__ void unary_kernel<uint64_t, SqrtOp>(const uint64_t *, uint64_t *, size_t, SqrtOp);  // kUInt64
template __global__ void unary_kernel<bool, SqrtOp>(const bool *, bool *, size_t, SqrtOp);              // kBool

template __global__ void unary_kernel<float, SquareOp>(const float *, float *, size_t, SquareOp);  // kFloat32
template __global__ void unary_kernel<double, SquareOp>(const double *,
                                                        double *,
                                                        size_t,
                                                        SquareOp);  // kFloat64/kDouble
template __global__ void unary_kernel<int8_t, SquareOp>(const int8_t *, int8_t *, size_t, SquareOp);        // kInt8
template __global__ void unary_kernel<int16_t, SquareOp>(const int16_t *, int16_t *, size_t, SquareOp);     // kInt16
template __global__ void unary_kernel<int32_t, SquareOp>(const int32_t *, int32_t *, size_t, SquareOp);     // kInt32
template __global__ void unary_kernel<int64_t, SquareOp>(const int64_t *, int64_t *, size_t, SquareOp);     // kInt64
template __global__ void unary_kernel<uint8_t, SquareOp>(const uint8_t *, uint8_t *, size_t, SquareOp);     // kUInt8
template __global__ void unary_kernel<uint16_t, SquareOp>(const uint16_t *, uint16_t *, size_t, SquareOp);  // kUInt16
template __global__ void unary_kernel<uint32_t, SquareOp>(const uint32_t *, uint32_t *, size_t, SquareOp);  // kUInt32
template __global__ void unary_kernel<uint64_t, SquareOp>(const uint64_t *, uint64_t *, size_t, SquareOp);  // kUInt64
template __global__ void unary_kernel<bool, SquareOp>(const bool *, bool *, size_t, SquareOp);              // kBool
template __global__ void unary_kernel<float, NegOp>(const float *, float *, size_t, NegOp);                 // kFloat32
template __global__ void unary_kernel<double, NegOp>(const double *, double *, size_t, NegOp);     // kFloat64/kDouble
template __global__ void unary_kernel<int8_t, NegOp>(const int8_t *, int8_t *, size_t, NegOp);     // kInt8
template __global__ void unary_kernel<int16_t, NegOp>(const int16_t *, int16_t *, size_t, NegOp);  // kInt16
template __global__ void unary_kernel<int32_t, NegOp>(const int32_t *, int32_t *, size_t, NegOp);  // kInt32
template __global__ void unary_kernel<int64_t, NegOp>(const int64_t *, int64_t *, size_t, NegOp);  // kInt64
template __global__ void unary_kernel<uint8_t, NegOp>(const uint8_t *, uint8_t *, size_t, NegOp);  // kUInt8
template __global__ void unary_kernel<uint16_t, NegOp>(const uint16_t *, uint16_t *, size_t, NegOp);  // kUInt16
template __global__ void unary_kernel<uint32_t, NegOp>(const uint32_t *, uint32_t *, size_t, NegOp);  // kUInt32
template __global__ void unary_kernel<uint64_t, NegOp>(const uint64_t *, uint64_t *, size_t, NegOp);  // kUInt64
template __global__ void unary_kernel<bool, NegOp>(const bool *, bool *, size_t, NegOp);              // kBool
template __global__ void unary_kernel<float, ReLUOp>(const float *, float *, size_t, ReLUOp);         // kFloat32
template __global__ void unary_kernel<double, ReLUOp>(const double *, double *, size_t, ReLUOp);     // kFloat64/kDouble
template __global__ void unary_kernel<int8_t, ReLUOp>(const int8_t *, int8_t *, size_t, ReLUOp);     // kInt8
template __global__ void unary_kernel<int16_t, ReLUOp>(const int16_t *, int16_t *, size_t, ReLUOp);  // kInt16
template __global__ void unary_kernel<int32_t, ReLUOp>(const int32_t *, int32_t *, size_t, ReLUOp);  // kInt32
template __global__ void unary_kernel<int64_t, ReLUOp>(const int64_t *, int64_t *, size_t, ReLUOp);  // kInt64
template __global__ void unary_kernel<uint8_t, ReLUOp>(const uint8_t *, uint8_t *, size_t, ReLUOp);  // kUInt8
template __global__ void unary_kernel<uint16_t, ReLUOp>(const uint16_t *, uint16_t *, size_t, ReLUOp);  // kUInt16
template __global__ void unary_kernel<uint32_t, ReLUOp>(const uint32_t *, uint32_t *, size_t, ReLUOp);  // kUInt32
template __global__ void unary_kernel<uint64_t, ReLUOp>(const uint64_t *, uint64_t *, size_t, ReLUOp);  // kUInt64
template __global__ void unary_kernel<bool, ReLUOp>(const bool *, bool *, size_t, ReLUOp);              // kBool

// 标量内核的显式实例化
// 按照 DataType 枚举顺序排列
template __global__ void scalar_kernel<float, AddOp>(const float *, float, float *, size_t, AddOp);  // kFloat32
template __global__ void scalar_kernel<double, AddOp>(const double *,
                                                      double,
                                                      double *,
                                                      size_t,
                                                      AddOp);  // kFloat64/kDouble
template __global__ void scalar_kernel<int8_t, AddOp>(const int8_t *, int8_t, int8_t *, size_t, AddOp);      // kInt8
template __global__ void scalar_kernel<int16_t, AddOp>(const int16_t *, int16_t, int16_t *, size_t, AddOp);  // kInt16
template __global__ void scalar_kernel<int32_t, AddOp>(const int32_t *, int32_t, int32_t *, size_t, AddOp);  // kInt32
template __global__ void scalar_kernel<int64_t, AddOp>(const int64_t *, int64_t, int64_t *, size_t, AddOp);  // kInt64
template __global__ void scalar_kernel<uint8_t, AddOp>(const uint8_t *, uint8_t, uint8_t *, size_t, AddOp);  // kUInt8
template __global__ void scalar_kernel<uint16_t, AddOp>(const uint16_t *,
                                                        uint16_t,
                                                        uint16_t *,
                                                        size_t,
                                                        AddOp);  // kUInt16
template __global__ void scalar_kernel<uint32_t, AddOp>(const uint32_t *,
                                                        uint32_t,
                                                        uint32_t *,
                                                        size_t,
                                                        AddOp);  // kUInt32
template __global__ void scalar_kernel<uint64_t, AddOp>(const uint64_t *,
                                                        uint64_t,
                                                        uint64_t *,
                                                        size_t,
                                                        AddOp);                                  // kUInt64
template __global__ void scalar_kernel<bool, AddOp>(const bool *, bool, bool *, size_t, AddOp);  // kBool

template __global__ void scalar_kernel<float, MultiplyOp>(const float *,
                                                          float,
                                                          float *,
                                                          size_t,
                                                          MultiplyOp);  // kFloat32
template __global__ void scalar_kernel<double, MultiplyOp>(const double *,
                                                           double,
                                                           double *,
                                                           size_t,
                                                           MultiplyOp);  // kFloat64/kDouble
template __global__ void scalar_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                           int8_t,
                                                           int8_t *,
                                                           size_t,
                                                           MultiplyOp);  // kInt8
template __global__ void scalar_kernel<int16_t, MultiplyOp>(const int16_t *,
                                                            int16_t,
                                                            int16_t *,
                                                            size_t,
                                                            MultiplyOp);  // kInt16
template __global__ void scalar_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                            int32_t,
                                                            int32_t *,
                                                            size_t,
                                                            MultiplyOp);  // kInt32
template __global__ void scalar_kernel<int64_t, MultiplyOp>(const int64_t *,
                                                            int64_t,
                                                            int64_t *,
                                                            size_t,
                                                            MultiplyOp);  // kInt64
template __global__ void scalar_kernel<uint8_t, MultiplyOp>(const uint8_t *,
                                                            uint8_t,
                                                            uint8_t *,
                                                            size_t,
                                                            MultiplyOp);  // kUInt8
template __global__ void scalar_kernel<uint16_t, MultiplyOp>(const uint16_t *,
                                                             uint16_t,
                                                             uint16_t *,
                                                             size_t,
                                                             MultiplyOp);  // kUInt16
template __global__ void scalar_kernel<uint32_t, MultiplyOp>(const uint32_t *,
                                                             uint32_t,
                                                             uint32_t *,
                                                             size_t,
                                                             MultiplyOp);  // kUInt32
template __global__ void scalar_kernel<uint64_t, MultiplyOp>(const uint64_t *,
                                                             uint64_t,
                                                             uint64_t *,
                                                             size_t,
                                                             MultiplyOp);                                  // kUInt64
template __global__ void scalar_kernel<bool, MultiplyOp>(const bool *, bool, bool *, size_t, MultiplyOp);  // kBool

// 启动函数的显式实例化
template void launch_elementwise_kernel<float, AddOp>(const float *,
                                                      const float *,
                                                      float *,
                                                      size_t,
                                                      AddOp,
                                                      cudaStream_t);
template void launch_elementwise_kernel<double, AddOp>(const double *,
                                                       const double *,
                                                       double *,
                                                       size_t,
                                                       AddOp,
                                                       cudaStream_t);
template void launch_elementwise_kernel<int32_t, AddOp>(const int32_t *,
                                                        const int32_t *,
                                                        int32_t *,
                                                        size_t,
                                                        AddOp,
                                                        cudaStream_t);
template void launch_elementwise_kernel<int8_t, AddOp>(const int8_t *,
                                                       const int8_t *,
                                                       int8_t *,
                                                       size_t,
                                                       AddOp,
                                                       cudaStream_t);
template void launch_elementwise_kernel<short, AddOp>(const short *,
                                                      const short *,
                                                      short *,
                                                      size_t,
                                                      AddOp,
                                                      cudaStream_t);
template void launch_elementwise_kernel<long, AddOp>(const long *, const long *, long *, size_t, AddOp, cudaStream_t);
template void launch_elementwise_kernel<unsigned char, AddOp>(const unsigned char *,
                                                              const unsigned char *,
                                                              unsigned char *,
                                                              size_t,
                                                              AddOp,
                                                              cudaStream_t);
template void launch_elementwise_kernel<unsigned short, AddOp>(const unsigned short *,
                                                               const unsigned short *,
                                                               unsigned short *,
                                                               size_t,
                                                               AddOp,
                                                               cudaStream_t);
template void launch_elementwise_kernel<unsigned int, AddOp>(const unsigned int *,
                                                             const unsigned int *,
                                                             unsigned int *,
                                                             size_t,
                                                             AddOp,
                                                             cudaStream_t);
template void launch_elementwise_kernel<unsigned long, AddOp>(const unsigned long *,
                                                              const unsigned long *,
                                                              unsigned long *,
                                                              size_t,
                                                              AddOp,
                                                              cudaStream_t);
template void launch_elementwise_kernel<bool, AddOp>(const bool *, const bool *, bool *, size_t, AddOp, cudaStream_t);

template void launch_elementwise_kernel<float, SubtractOp>(const float *,
                                                           const float *,
                                                           float *,
                                                           size_t,
                                                           SubtractOp,
                                                           cudaStream_t);
template void launch_elementwise_kernel<double, SubtractOp>(const double *,
                                                            const double *,
                                                            double *,
                                                            size_t,
                                                            SubtractOp,
                                                            cudaStream_t);
template void launch_elementwise_kernel<int32_t, SubtractOp>(const int32_t *,
                                                             const int32_t *,
                                                             int32_t *,
                                                             size_t,
                                                             SubtractOp,
                                                             cudaStream_t);
template void launch_elementwise_kernel<int8_t, SubtractOp>(const int8_t *,
                                                            const int8_t *,
                                                            int8_t *,
                                                            size_t,
                                                            SubtractOp,
                                                            cudaStream_t);
template void launch_elementwise_kernel<short, SubtractOp>(const short *,
                                                           const short *,
                                                           short *,
                                                           size_t,
                                                           SubtractOp,
                                                           cudaStream_t);
template void launch_elementwise_kernel<long, SubtractOp>(const long *,
                                                          const long *,
                                                          long *,
                                                          size_t,
                                                          SubtractOp,
                                                          cudaStream_t);
template void launch_elementwise_kernel<unsigned char, SubtractOp>(const unsigned char *,
                                                                   const unsigned char *,
                                                                   unsigned char *,
                                                                   size_t,
                                                                   SubtractOp,
                                                                   cudaStream_t);
template void launch_elementwise_kernel<unsigned short, SubtractOp>(const unsigned short *,
                                                                    const unsigned short *,
                                                                    unsigned short *,
                                                                    size_t,
                                                                    SubtractOp,
                                                                    cudaStream_t);
template void launch_elementwise_kernel<unsigned int, SubtractOp>(const unsigned int *,
                                                                  const unsigned int *,
                                                                  unsigned int *,
                                                                  size_t,
                                                                  SubtractOp,
                                                                  cudaStream_t);
template void launch_elementwise_kernel<unsigned long, SubtractOp>(const unsigned long *,
                                                                   const unsigned long *,
                                                                   unsigned long *,
                                                                   size_t,
                                                                   SubtractOp,
                                                                   cudaStream_t);
template void launch_elementwise_kernel<bool, SubtractOp>(const bool *,
                                                          const bool *,
                                                          bool *,
                                                          size_t,
                                                          SubtractOp,
                                                          cudaStream_t);

template void launch_elementwise_kernel<float, MultiplyOp>(const float *,
                                                           const float *,
                                                           float *,
                                                           size_t,
                                                           MultiplyOp,
                                                           cudaStream_t);
template void launch_elementwise_kernel<double, MultiplyOp>(const double *,
                                                            const double *,
                                                            double *,
                                                            size_t,
                                                            MultiplyOp,
                                                            cudaStream_t);
template void launch_elementwise_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                             const int32_t *,
                                                             int32_t *,
                                                             size_t,
                                                             MultiplyOp,
                                                             cudaStream_t);
template void launch_elementwise_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                            const int8_t *,
                                                            int8_t *,
                                                            size_t,
                                                            MultiplyOp,
                                                            cudaStream_t);
template void launch_elementwise_kernel<short, MultiplyOp>(const short *,
                                                           const short *,
                                                           short *,
                                                           size_t,
                                                           MultiplyOp,
                                                           cudaStream_t);
template void launch_elementwise_kernel<long, MultiplyOp>(const long *,
                                                          const long *,
                                                          long *,
                                                          size_t,
                                                          MultiplyOp,
                                                          cudaStream_t);
template void launch_elementwise_kernel<unsigned char, MultiplyOp>(const unsigned char *,
                                                                   const unsigned char *,
                                                                   unsigned char *,
                                                                   size_t,
                                                                   MultiplyOp,
                                                                   cudaStream_t);
template void launch_elementwise_kernel<unsigned short, MultiplyOp>(const unsigned short *,
                                                                    const unsigned short *,
                                                                    unsigned short *,
                                                                    size_t,
                                                                    MultiplyOp,
                                                                    cudaStream_t);
template void launch_elementwise_kernel<unsigned int, MultiplyOp>(const unsigned int *,
                                                                  const unsigned int *,
                                                                  unsigned int *,
                                                                  size_t,
                                                                  MultiplyOp,
                                                                  cudaStream_t);
template void launch_elementwise_kernel<unsigned long, MultiplyOp>(const unsigned long *,
                                                                   const unsigned long *,
                                                                   unsigned long *,
                                                                   size_t,
                                                                   MultiplyOp,
                                                                   cudaStream_t);
template void launch_elementwise_kernel<bool, MultiplyOp>(const bool *,
                                                          const bool *,
                                                          bool *,
                                                          size_t,
                                                          MultiplyOp,
                                                          cudaStream_t);

template void launch_elementwise_kernel<float, DivideOp>(const float *,
                                                         const float *,
                                                         float *,
                                                         size_t,
                                                         DivideOp,
                                                         cudaStream_t);
template void launch_elementwise_kernel<double, DivideOp>(const double *,
                                                          const double *,
                                                          double *,
                                                          size_t,
                                                          DivideOp,
                                                          cudaStream_t);
template void launch_elementwise_kernel<int32_t, DivideOp>(const int32_t *,
                                                           const int32_t *,
                                                           int32_t *,
                                                           size_t,
                                                           DivideOp,
                                                           cudaStream_t);
template void launch_elementwise_kernel<int8_t, DivideOp>(const int8_t *,
                                                          const int8_t *,
                                                          int8_t *,
                                                          size_t,
                                                          DivideOp,
                                                          cudaStream_t);
template void launch_elementwise_kernel<short, DivideOp>(const short *,
                                                         const short *,
                                                         short *,
                                                         size_t,
                                                         DivideOp,
                                                         cudaStream_t);
template void launch_elementwise_kernel<long, DivideOp>(const long *,
                                                        const long *,
                                                        long *,
                                                        size_t,
                                                        DivideOp,
                                                        cudaStream_t);
template void launch_elementwise_kernel<unsigned char, DivideOp>(const unsigned char *,
                                                                 const unsigned char *,
                                                                 unsigned char *,
                                                                 size_t,
                                                                 DivideOp,
                                                                 cudaStream_t);
template void launch_elementwise_kernel<unsigned short, DivideOp>(const unsigned short *,
                                                                  const unsigned short *,
                                                                  unsigned short *,
                                                                  size_t,
                                                                  DivideOp,
                                                                  cudaStream_t);
template void launch_elementwise_kernel<unsigned int, DivideOp>(const unsigned int *,
                                                                const unsigned int *,
                                                                unsigned int *,
                                                                size_t,
                                                                DivideOp,
                                                                cudaStream_t);
template void launch_elementwise_kernel<unsigned long, DivideOp>(const unsigned long *,
                                                                 const unsigned long *,
                                                                 unsigned long *,
                                                                 size_t,
                                                                 DivideOp,
                                                                 cudaStream_t);
template void launch_elementwise_kernel<bool, DivideOp>(const bool *,
                                                        const bool *,
                                                        bool *,
                                                        size_t,
                                                        DivideOp,
                                                        cudaStream_t);

// 一元内核的显式实例化 - 按照 DataType 枚举顺序排列
// ExpOp 实例化
template void launch_unary_kernel<float, ExpOp>(const float *, float *, size_t, ExpOp, cudaStream_t);  // kFloat32
template void launch_unary_kernel<double, ExpOp>(const double *,
                                                 double *,
                                                 size_t,
                                                 ExpOp,
                                                 cudaStream_t);  // kFloat64/kDouble
template void launch_unary_kernel<int8_t, ExpOp>(const int8_t *, int8_t *, size_t, ExpOp, cudaStream_t);     // kInt8
template void launch_unary_kernel<int16_t, ExpOp>(const int16_t *, int16_t *, size_t, ExpOp, cudaStream_t);  // kInt16
template void launch_unary_kernel<int32_t, ExpOp>(const int32_t *, int32_t *, size_t, ExpOp, cudaStream_t);  // kInt32
template void launch_unary_kernel<int64_t, ExpOp>(const int64_t *, int64_t *, size_t, ExpOp, cudaStream_t);  // kInt64
template void launch_unary_kernel<uint8_t, ExpOp>(const uint8_t *, uint8_t *, size_t, ExpOp, cudaStream_t);  // kUInt8
template void launch_unary_kernel<uint16_t, ExpOp>(const uint16_t *,
                                                   uint16_t *,
                                                   size_t,
                                                   ExpOp,
                                                   cudaStream_t);  // kUInt16
template void launch_unary_kernel<uint32_t, ExpOp>(const uint32_t *,
                                                   uint32_t *,
                                                   size_t,
                                                   ExpOp,
                                                   cudaStream_t);  // kUInt32
template void launch_unary_kernel<uint64_t, ExpOp>(const uint64_t *,
                                                   uint64_t *,
                                                   size_t,
                                                   ExpOp,
                                                   cudaStream_t);                                   // kUInt64
template void launch_unary_kernel<bool, ExpOp>(const bool *, bool *, size_t, ExpOp, cudaStream_t);  // kBool

// LogOp 实例化
template void launch_unary_kernel<float, LogOp>(const float *, float *, size_t, LogOp, cudaStream_t);  // kFloat32
template void launch_unary_kernel<double, LogOp>(const double *,
                                                 double *,
                                                 size_t,
                                                 LogOp,
                                                 cudaStream_t);  // kFloat64/kDouble
template void launch_unary_kernel<int8_t, LogOp>(const int8_t *, int8_t *, size_t, LogOp, cudaStream_t);     // kInt8
template void launch_unary_kernel<int16_t, LogOp>(const int16_t *, int16_t *, size_t, LogOp, cudaStream_t);  // kInt16
template void launch_unary_kernel<int32_t, LogOp>(const int32_t *, int32_t *, size_t, LogOp, cudaStream_t);  // kInt32
template void launch_unary_kernel<int64_t, LogOp>(const int64_t *, int64_t *, size_t, LogOp, cudaStream_t);  // kInt64
template void launch_unary_kernel<uint8_t, LogOp>(const uint8_t *, uint8_t *, size_t, LogOp, cudaStream_t);  // kUInt8
template void launch_unary_kernel<uint16_t, LogOp>(const uint16_t *,
                                                   uint16_t *,
                                                   size_t,
                                                   LogOp,
                                                   cudaStream_t);  // kUInt16
template void launch_unary_kernel<uint32_t, LogOp>(const uint32_t *,
                                                   uint32_t *,
                                                   size_t,
                                                   LogOp,
                                                   cudaStream_t);  // kUInt32
template void launch_unary_kernel<uint64_t, LogOp>(const uint64_t *,
                                                   uint64_t *,
                                                   size_t,
                                                   LogOp,
                                                   cudaStream_t);                                   // kUInt64
template void launch_unary_kernel<bool, LogOp>(const bool *, bool *, size_t, LogOp, cudaStream_t);  // kBool

// SqrtOp 实例化
template void launch_unary_kernel<float, SqrtOp>(const float *, float *, size_t, SqrtOp, cudaStream_t);  // kFloat32
template void launch_unary_kernel<double, SqrtOp>(const double *,
                                                  double *,
                                                  size_t,
                                                  SqrtOp,
                                                  cudaStream_t);  // kFloat64/kDouble
template void launch_unary_kernel<int8_t, SqrtOp>(const int8_t *, int8_t *, size_t, SqrtOp, cudaStream_t);     // kInt8
template void launch_unary_kernel<int16_t, SqrtOp>(const int16_t *, int16_t *, size_t, SqrtOp, cudaStream_t);  // kInt16
template void launch_unary_kernel<int32_t, SqrtOp>(const int32_t *, int32_t *, size_t, SqrtOp, cudaStream_t);  // kInt32
template void launch_unary_kernel<int64_t, SqrtOp>(const int64_t *, int64_t *, size_t, SqrtOp, cudaStream_t);  // kInt64
template void launch_unary_kernel<uint8_t, SqrtOp>(const uint8_t *, uint8_t *, size_t, SqrtOp, cudaStream_t);  // kUInt8
template void launch_unary_kernel<uint16_t, SqrtOp>(const uint16_t *,
                                                    uint16_t *,
                                                    size_t,
                                                    SqrtOp,
                                                    cudaStream_t);  // kUInt16
template void launch_unary_kernel<uint32_t, SqrtOp>(const uint32_t *,
                                                    uint32_t *,
                                                    size_t,
                                                    SqrtOp,
                                                    cudaStream_t);  // kUInt32
template void launch_unary_kernel<uint64_t, SqrtOp>(const uint64_t *,
                                                    uint64_t *,
                                                    size_t,
                                                    SqrtOp,
                                                    cudaStream_t);                                    // kUInt64
template void launch_unary_kernel<bool, SqrtOp>(const bool *, bool *, size_t, SqrtOp, cudaStream_t);  // kBool

// SquareOp 实例化
template void launch_unary_kernel<float, SquareOp>(const float *, float *, size_t, SquareOp, cudaStream_t);  // kFloat32
template void launch_unary_kernel<double, SquareOp>(const double *,
                                                    double *,
                                                    size_t,
                                                    SquareOp,
                                                    cudaStream_t);  // kFloat64/kDouble
template void launch_unary_kernel<int8_t, SquareOp>(const int8_t *, int8_t *, size_t, SquareOp, cudaStream_t);  // kInt8
template void launch_unary_kernel<int16_t, SquareOp>(const int16_t *,
                                                     int16_t *,
                                                     size_t,
                                                     SquareOp,
                                                     cudaStream_t);  // kInt16
template void launch_unary_kernel<int32_t, SquareOp>(const int32_t *,
                                                     int32_t *,
                                                     size_t,
                                                     SquareOp,
                                                     cudaStream_t);  // kInt32
template void launch_unary_kernel<int64_t, SquareOp>(const int64_t *,
                                                     int64_t *,
                                                     size_t,
                                                     SquareOp,
                                                     cudaStream_t);  // kInt64
template void launch_unary_kernel<uint8_t, SquareOp>(const uint8_t *,
                                                     uint8_t *,
                                                     size_t,
                                                     SquareOp,
                                                     cudaStream_t);  // kUInt8
template void launch_unary_kernel<uint16_t, SquareOp>(const uint16_t *,
                                                      uint16_t *,
                                                      size_t,
                                                      SquareOp,
                                                      cudaStream_t);  // kUInt16
template void launch_unary_kernel<uint32_t, SquareOp>(const uint32_t *,
                                                      uint32_t *,
                                                      size_t,
                                                      SquareOp,
                                                      cudaStream_t);  // kUInt32
template void launch_unary_kernel<uint64_t, SquareOp>(const uint64_t *,
                                                      uint64_t *,
                                                      size_t,
                                                      SquareOp,
                                                      cudaStream_t);                                      // kUInt64
template void launch_unary_kernel<bool, SquareOp>(const bool *, bool *, size_t, SquareOp, cudaStream_t);  // kBool

// NegOp 实例化
template void launch_unary_kernel<float, NegOp>(const float *, float *, size_t, NegOp, cudaStream_t);  // kFloat32
template void launch_unary_kernel<double, NegOp>(const double *,
                                                 double *,
                                                 size_t,
                                                 NegOp,
                                                 cudaStream_t);  // kFloat64/kDouble
template void launch_unary_kernel<int8_t, NegOp>(const int8_t *, int8_t *, size_t, NegOp, cudaStream_t);     // kInt8
template void launch_unary_kernel<int16_t, NegOp>(const int16_t *, int16_t *, size_t, NegOp, cudaStream_t);  // kInt16
template void launch_unary_kernel<int32_t, NegOp>(const int32_t *, int32_t *, size_t, NegOp, cudaStream_t);  // kInt32
template void launch_unary_kernel<int64_t, NegOp>(const int64_t *, int64_t *, size_t, NegOp, cudaStream_t);  // kInt64
template void launch_unary_kernel<uint8_t, NegOp>(const uint8_t *, uint8_t *, size_t, NegOp, cudaStream_t);  // kUInt8
template void launch_unary_kernel<uint16_t, NegOp>(const uint16_t *,
                                                   uint16_t *,
                                                   size_t,
                                                   NegOp,
                                                   cudaStream_t);  // kUInt16
template void launch_unary_kernel<uint32_t, NegOp>(const uint32_t *,
                                                   uint32_t *,
                                                   size_t,
                                                   NegOp,
                                                   cudaStream_t);  // kUInt32
template void launch_unary_kernel<uint64_t, NegOp>(const uint64_t *,
                                                   uint64_t *,
                                                   size_t,
                                                   NegOp,
                                                   cudaStream_t);                                   // kUInt64
template void launch_unary_kernel<bool, NegOp>(const bool *, bool *, size_t, NegOp, cudaStream_t);  // kBool

// ReLUOp 实例化
template void launch_unary_kernel<float, ReLUOp>(const float *, float *, size_t, ReLUOp, cudaStream_t);  // kFloat32
template void launch_unary_kernel<double, ReLUOp>(const double *,
                                                  double *,
                                                  size_t,
                                                  ReLUOp,
                                                  cudaStream_t);  // kFloat64/kDouble
template void launch_unary_kernel<int8_t, ReLUOp>(const int8_t *, int8_t *, size_t, ReLUOp, cudaStream_t);     // kInt8
template void launch_unary_kernel<int16_t, ReLUOp>(const int16_t *, int16_t *, size_t, ReLUOp, cudaStream_t);  // kInt16
template void launch_unary_kernel<int32_t, ReLUOp>(const int32_t *, int32_t *, size_t, ReLUOp, cudaStream_t);  // kInt32
template void launch_unary_kernel<int64_t, ReLUOp>(const int64_t *, int64_t *, size_t, ReLUOp, cudaStream_t);  // kInt64
template void launch_unary_kernel<uint8_t, ReLUOp>(const uint8_t *, uint8_t *, size_t, ReLUOp, cudaStream_t);  // kUInt8
template void launch_unary_kernel<uint16_t, ReLUOp>(const uint16_t *,
                                                    uint16_t *,
                                                    size_t,
                                                    ReLUOp,
                                                    cudaStream_t);  // kUInt16
template void launch_unary_kernel<uint32_t, ReLUOp>(const uint32_t *,
                                                    uint32_t *,
                                                    size_t,
                                                    ReLUOp,
                                                    cudaStream_t);  // kUInt32
template void launch_unary_kernel<uint64_t, ReLUOp>(const uint64_t *,
                                                    uint64_t *,
                                                    size_t,
                                                    ReLUOp,
                                                    cudaStream_t);                                    // kUInt64
template void launch_unary_kernel<bool, ReLUOp>(const bool *, bool *, size_t, ReLUOp, cudaStream_t);  // kBool

template void launch_scalar_kernel<float, AddOp>(const float *, float, float *, size_t, AddOp, cudaStream_t);
template void launch_scalar_kernel<double, AddOp>(const double *, double, double *, size_t, AddOp, cudaStream_t);
template void launch_scalar_kernel<int32_t, AddOp>(const int32_t *, int32_t, int32_t *, size_t, AddOp, cudaStream_t);
template void launch_scalar_kernel<int8_t, AddOp>(const int8_t *, int8_t, int8_t *, size_t, AddOp, cudaStream_t);
template void launch_scalar_kernel<short, AddOp>(const short *, short, short *, size_t, AddOp, cudaStream_t);
template void launch_scalar_kernel<long, AddOp>(const long *, long, long *, size_t, AddOp, cudaStream_t);
template void launch_scalar_kernel<unsigned char, AddOp>(const unsigned char *,
                                                         unsigned char,
                                                         unsigned char *,
                                                         size_t,
                                                         AddOp,
                                                         cudaStream_t);
template void launch_scalar_kernel<unsigned short, AddOp>(const unsigned short *,
                                                          unsigned short,
                                                          unsigned short *,
                                                          size_t,
                                                          AddOp,
                                                          cudaStream_t);
template void launch_scalar_kernel<unsigned int, AddOp>(const unsigned int *,
                                                        unsigned int,
                                                        unsigned int *,
                                                        size_t,
                                                        AddOp,
                                                        cudaStream_t);
template void launch_scalar_kernel<unsigned long, AddOp>(const unsigned long *,
                                                         unsigned long,
                                                         unsigned long *,
                                                         size_t,
                                                         AddOp,
                                                         cudaStream_t);
template void launch_scalar_kernel<bool, AddOp>(const bool *, bool, bool *, size_t, AddOp, cudaStream_t);

template void launch_scalar_kernel<float, MultiplyOp>(const float *, float, float *, size_t, MultiplyOp, cudaStream_t);
template void launch_scalar_kernel<double, MultiplyOp>(const double *,
                                                       double,
                                                       double *,
                                                       size_t,
                                                       MultiplyOp,
                                                       cudaStream_t);
template void launch_scalar_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                        int32_t,
                                                        int32_t *,
                                                        size_t,
                                                        MultiplyOp,
                                                        cudaStream_t);
template void launch_scalar_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                       int8_t,
                                                       int8_t *,
                                                       size_t,
                                                       MultiplyOp,
                                                       cudaStream_t);
template void launch_scalar_kernel<short, MultiplyOp>(const short *, short, short *, size_t, MultiplyOp, cudaStream_t);
template void launch_scalar_kernel<long, MultiplyOp>(const long *, long, long *, size_t, MultiplyOp, cudaStream_t);
template void launch_scalar_kernel<unsigned char, MultiplyOp>(const unsigned char *,
                                                              unsigned char,
                                                              unsigned char *,
                                                              size_t,
                                                              MultiplyOp,
                                                              cudaStream_t);
template void launch_scalar_kernel<unsigned short, MultiplyOp>(const unsigned short *,
                                                               unsigned short,
                                                               unsigned short *,
                                                               size_t,
                                                               MultiplyOp,
                                                               cudaStream_t);
template void launch_scalar_kernel<unsigned int, MultiplyOp>(const unsigned int *,
                                                             unsigned int,
                                                             unsigned int *,
                                                             size_t,
                                                             MultiplyOp,
                                                             cudaStream_t);
template void launch_scalar_kernel<unsigned long, MultiplyOp>(const unsigned long *,
                                                              unsigned long,
                                                              unsigned long *,
                                                              size_t,
                                                              MultiplyOp,
                                                              cudaStream_t);
template void launch_scalar_kernel<bool, MultiplyOp>(const bool *, bool, bool *, size_t, MultiplyOp, cudaStream_t);

// 简单广播内核的显式实例化
template __global__ void
simple_broadcast_kernel<float, AddOp>(const float *, const float *, float *, size_t, size_t, size_t, AddOp);
template __global__ void
simple_broadcast_kernel<double, AddOp>(const double *, const double *, double *, size_t, size_t, size_t, AddOp);
template __global__ void
simple_broadcast_kernel<int32_t, AddOp>(const int32_t *, const int32_t *, int32_t *, size_t, size_t, size_t, AddOp);
template __global__ void
simple_broadcast_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, AddOp);

template __global__ void
simple_broadcast_kernel<float, SubtractOp>(const float *, const float *, float *, size_t, size_t, size_t, SubtractOp);
template __global__ void simple_broadcast_kernel<double, SubtractOp>(const double *,
                                                                     const double *,
                                                                     double *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     SubtractOp);
template __global__ void simple_broadcast_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                      const int32_t *,
                                                                      int32_t *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      SubtractOp);
template __global__ void simple_broadcast_kernel<int8_t, SubtractOp>(const int8_t *,
                                                                     const int8_t *,
                                                                     int8_t *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     SubtractOp);

template __global__ void
simple_broadcast_kernel<float, MultiplyOp>(const float *, const float *, float *, size_t, size_t, size_t, MultiplyOp);
template __global__ void simple_broadcast_kernel<double, MultiplyOp>(const double *,
                                                                     const double *,
                                                                     double *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     MultiplyOp);
template __global__ void simple_broadcast_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                      const int32_t *,
                                                                      int32_t *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      MultiplyOp);
template __global__ void simple_broadcast_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                                     const int8_t *,
                                                                     int8_t *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     MultiplyOp);

template __global__ void
simple_broadcast_kernel<float, DivideOp>(const float *, const float *, float *, size_t, size_t, size_t, DivideOp);
template __global__ void
simple_broadcast_kernel<double, DivideOp>(const double *, const double *, double *, size_t, size_t, size_t, DivideOp);
template __global__ void simple_broadcast_kernel<int32_t, DivideOp>(const int32_t *,
                                                                    const int32_t *,
                                                                    int32_t *,
                                                                    size_t,
                                                                    size_t,
                                                                    size_t,
                                                                    DivideOp);
template __global__ void
simple_broadcast_kernel<int8_t, DivideOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, DivideOp);

// 添加缺失的数据类型实例化

// 标量广播内核的缺失实例化
template __global__ void
simple_broadcast_kernel<short, AddOp>(const short *, const short *, short *, size_t, size_t, size_t, AddOp);
template __global__ void
simple_broadcast_kernel<long, AddOp>(const long *, const long *, long *, size_t, size_t, size_t, AddOp);
template __global__ void simple_broadcast_kernel<unsigned char, AddOp>(const unsigned char *,
                                                                       const unsigned char *,
                                                                       unsigned char *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       AddOp);
template __global__ void simple_broadcast_kernel<unsigned short, AddOp>(const unsigned short *,
                                                                        const unsigned short *,
                                                                        unsigned short *,
                                                                        size_t,
                                                                        size_t,
                                                                        size_t,
                                                                        AddOp);
template __global__ void simple_broadcast_kernel<unsigned int, AddOp>(const unsigned int *,
                                                                      const unsigned int *,
                                                                      unsigned int *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      AddOp);
template __global__ void simple_broadcast_kernel<unsigned long, AddOp>(const unsigned long *,
                                                                       const unsigned long *,
                                                                       unsigned long *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       AddOp);
template __global__ void
simple_broadcast_kernel<bool, AddOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, AddOp);

template __global__ void
simple_broadcast_kernel<short, SubtractOp>(const short *, const short *, short *, size_t, size_t, size_t, SubtractOp);
template __global__ void
simple_broadcast_kernel<long, SubtractOp>(const long *, const long *, long *, size_t, size_t, size_t, SubtractOp);
template __global__ void simple_broadcast_kernel<unsigned char, SubtractOp>(const unsigned char *,
                                                                            const unsigned char *,
                                                                            unsigned char *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            SubtractOp);
template __global__ void simple_broadcast_kernel<unsigned short, SubtractOp>(const unsigned short *,
                                                                             const unsigned short *,
                                                                             unsigned short *,
                                                                             size_t,
                                                                             size_t,
                                                                             size_t,
                                                                             SubtractOp);
template __global__ void simple_broadcast_kernel<unsigned int, SubtractOp>(const unsigned int *,
                                                                           const unsigned int *,
                                                                           unsigned int *,
                                                                           size_t,
                                                                           size_t,
                                                                           size_t,
                                                                           SubtractOp);
template __global__ void simple_broadcast_kernel<unsigned long, SubtractOp>(const unsigned long *,
                                                                            const unsigned long *,
                                                                            unsigned long *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            SubtractOp);
template __global__ void
simple_broadcast_kernel<bool, SubtractOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, SubtractOp);

template __global__ void
simple_broadcast_kernel<short, MultiplyOp>(const short *, const short *, short *, size_t, size_t, size_t, MultiplyOp);
template __global__ void
simple_broadcast_kernel<long, MultiplyOp>(const long *, const long *, long *, size_t, size_t, size_t, MultiplyOp);
template __global__ void simple_broadcast_kernel<unsigned char, MultiplyOp>(const unsigned char *,
                                                                            const unsigned char *,
                                                                            unsigned char *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            MultiplyOp);
template __global__ void simple_broadcast_kernel<unsigned short, MultiplyOp>(const unsigned short *,
                                                                             const unsigned short *,
                                                                             unsigned short *,
                                                                             size_t,
                                                                             size_t,
                                                                             size_t,
                                                                             MultiplyOp);
template __global__ void simple_broadcast_kernel<unsigned int, MultiplyOp>(const unsigned int *,
                                                                           const unsigned int *,
                                                                           unsigned int *,
                                                                           size_t,
                                                                           size_t,
                                                                           size_t,
                                                                           MultiplyOp);
template __global__ void simple_broadcast_kernel<unsigned long, MultiplyOp>(const unsigned long *,
                                                                            const unsigned long *,
                                                                            unsigned long *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            MultiplyOp);
template __global__ void
simple_broadcast_kernel<bool, MultiplyOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, MultiplyOp);

template __global__ void
simple_broadcast_kernel<short, DivideOp>(const short *, const short *, short *, size_t, size_t, size_t, DivideOp);
template __global__ void
simple_broadcast_kernel<long, DivideOp>(const long *, const long *, long *, size_t, size_t, size_t, DivideOp);
template __global__ void simple_broadcast_kernel<unsigned char, DivideOp>(const unsigned char *,
                                                                          const unsigned char *,
                                                                          unsigned char *,
                                                                          size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          DivideOp);
template __global__ void simple_broadcast_kernel<unsigned short, DivideOp>(const unsigned short *,
                                                                           const unsigned short *,
                                                                           unsigned short *,
                                                                           size_t,
                                                                           size_t,
                                                                           size_t,
                                                                           DivideOp);
template __global__ void simple_broadcast_kernel<unsigned int, DivideOp>(const unsigned int *,
                                                                         const unsigned int *,
                                                                         unsigned int *,
                                                                         size_t,
                                                                         size_t,
                                                                         size_t,
                                                                         DivideOp);
template __global__ void simple_broadcast_kernel<unsigned long, DivideOp>(const unsigned long *,
                                                                          const unsigned long *,
                                                                          unsigned long *,
                                                                          size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          DivideOp);
template __global__ void
simple_broadcast_kernel<bool, DivideOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, DivideOp);

// 简单广播启动函数的显式实例化
template void launch_simple_broadcast_kernel<float, AddOp>(const float *,
                                                           const float *,
                                                           float *,
                                                           size_t,
                                                           size_t,
                                                           size_t,
                                                           AddOp,
                                                           cudaStream_t);
template void launch_simple_broadcast_kernel<double, AddOp>(const double *,
                                                            const double *,
                                                            double *,
                                                            size_t,
                                                            size_t,
                                                            size_t,
                                                            AddOp,
                                                            cudaStream_t);
template void launch_simple_broadcast_kernel<int32_t, AddOp>(const int32_t *,
                                                             const int32_t *,
                                                             int32_t *,
                                                             size_t,
                                                             size_t,
                                                             size_t,
                                                             AddOp,
                                                             cudaStream_t);
template void launch_simple_broadcast_kernel<int8_t, AddOp>(const int8_t *,
                                                            const int8_t *,
                                                            int8_t *,
                                                            size_t,
                                                            size_t,
                                                            size_t,
                                                            AddOp,
                                                            cudaStream_t);
template void launch_simple_broadcast_kernel<short, AddOp>(const short *,
                                                           const short *,
                                                           short *,
                                                           size_t,
                                                           size_t,
                                                           size_t,
                                                           AddOp,
                                                           cudaStream_t);
template void launch_simple_broadcast_kernel<long, AddOp>(const long *,
                                                          const long *,
                                                          long *,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          AddOp,
                                                          cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned char, AddOp>(const unsigned char *,
                                                                   const unsigned char *,
                                                                   unsigned char *,
                                                                   size_t,
                                                                   size_t,
                                                                   size_t,
                                                                   AddOp,
                                                                   cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned short, AddOp>(const unsigned short *,
                                                                    const unsigned short *,
                                                                    unsigned short *,
                                                                    size_t,
                                                                    size_t,
                                                                    size_t,
                                                                    AddOp,
                                                                    cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned int, AddOp>(const unsigned int *,
                                                                  const unsigned int *,
                                                                  unsigned int *,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  AddOp,
                                                                  cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned long, AddOp>(const unsigned long *,
                                                                   const unsigned long *,
                                                                   unsigned long *,
                                                                   size_t,
                                                                   size_t,
                                                                   size_t,
                                                                   AddOp,
                                                                   cudaStream_t);
template void launch_simple_broadcast_kernel<bool, AddOp>(const bool *,
                                                          const bool *,
                                                          bool *,
                                                          size_t,
                                                          size_t,
                                                          size_t,
                                                          AddOp,
                                                          cudaStream_t);

template void launch_simple_broadcast_kernel<float, SubtractOp>(const float *,
                                                                const float *,
                                                                float *,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                SubtractOp,
                                                                cudaStream_t);
template void launch_simple_broadcast_kernel<double, SubtractOp>(const double *,
                                                                 const double *,
                                                                 double *,
                                                                 size_t,
                                                                 size_t,
                                                                 size_t,
                                                                 SubtractOp,
                                                                 cudaStream_t);
template void launch_simple_broadcast_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                  const int32_t *,
                                                                  int32_t *,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  SubtractOp,
                                                                  cudaStream_t);
template void launch_simple_broadcast_kernel<int8_t, SubtractOp>(const int8_t *,
                                                                 const int8_t *,
                                                                 int8_t *,
                                                                 size_t,
                                                                 size_t,
                                                                 size_t,
                                                                 SubtractOp,
                                                                 cudaStream_t);
template void launch_simple_broadcast_kernel<short, SubtractOp>(const short *,
                                                                const short *,
                                                                short *,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                SubtractOp,
                                                                cudaStream_t);
template void launch_simple_broadcast_kernel<long, SubtractOp>(const long *,
                                                               const long *,
                                                               long *,
                                                               size_t,
                                                               size_t,
                                                               size_t,
                                                               SubtractOp,
                                                               cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned char, SubtractOp>(const unsigned char *,
                                                                        const unsigned char *,
                                                                        unsigned char *,
                                                                        size_t,
                                                                        size_t,
                                                                        size_t,
                                                                        SubtractOp,
                                                                        cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned short, SubtractOp>(const unsigned short *,
                                                                         const unsigned short *,
                                                                         unsigned short *,
                                                                         size_t,
                                                                         size_t,
                                                                         size_t,
                                                                         SubtractOp,
                                                                         cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned int, SubtractOp>(const unsigned int *,
                                                                       const unsigned int *,
                                                                       unsigned int *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       SubtractOp,
                                                                       cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned long, SubtractOp>(const unsigned long *,
                                                                        const unsigned long *,
                                                                        unsigned long *,
                                                                        size_t,
                                                                        size_t,
                                                                        size_t,
                                                                        SubtractOp,
                                                                        cudaStream_t);
template void launch_simple_broadcast_kernel<bool, SubtractOp>(const bool *,
                                                               const bool *,
                                                               bool *,
                                                               size_t,
                                                               size_t,
                                                               size_t,
                                                               SubtractOp,
                                                               cudaStream_t);

template void launch_simple_broadcast_kernel<float, MultiplyOp>(const float *,
                                                                const float *,
                                                                float *,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                MultiplyOp,
                                                                cudaStream_t);
template void launch_simple_broadcast_kernel<double, MultiplyOp>(const double *,
                                                                 const double *,
                                                                 double *,
                                                                 size_t,
                                                                 size_t,
                                                                 size_t,
                                                                 MultiplyOp,
                                                                 cudaStream_t);
template void launch_simple_broadcast_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                  const int32_t *,
                                                                  int32_t *,
                                                                  size_t,
                                                                  size_t,
                                                                  size_t,
                                                                  MultiplyOp,
                                                                  cudaStream_t);
template void launch_simple_broadcast_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                                 const int8_t *,
                                                                 int8_t *,
                                                                 size_t,
                                                                 size_t,
                                                                 size_t,
                                                                 MultiplyOp,
                                                                 cudaStream_t);
template void launch_simple_broadcast_kernel<short, MultiplyOp>(const short *,
                                                                const short *,
                                                                short *,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                MultiplyOp,
                                                                cudaStream_t);
template void launch_simple_broadcast_kernel<long, MultiplyOp>(const long *,
                                                               const long *,
                                                               long *,
                                                               size_t,
                                                               size_t,
                                                               size_t,
                                                               MultiplyOp,
                                                               cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned char, MultiplyOp>(const unsigned char *,
                                                                        const unsigned char *,
                                                                        unsigned char *,
                                                                        size_t,
                                                                        size_t,
                                                                        size_t,
                                                                        MultiplyOp,
                                                                        cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned short, MultiplyOp>(const unsigned short *,
                                                                         const unsigned short *,
                                                                         unsigned short *,
                                                                         size_t,
                                                                         size_t,
                                                                         size_t,
                                                                         MultiplyOp,
                                                                         cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned int, MultiplyOp>(const unsigned int *,
                                                                       const unsigned int *,
                                                                       unsigned int *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       MultiplyOp,
                                                                       cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned long, MultiplyOp>(const unsigned long *,
                                                                        const unsigned long *,
                                                                        unsigned long *,
                                                                        size_t,
                                                                        size_t,
                                                                        size_t,
                                                                        MultiplyOp,
                                                                        cudaStream_t);
template void launch_simple_broadcast_kernel<bool, MultiplyOp>(const bool *,
                                                               const bool *,
                                                               bool *,
                                                               size_t,
                                                               size_t,
                                                               size_t,
                                                               MultiplyOp,
                                                               cudaStream_t);

template void launch_simple_broadcast_kernel<float, DivideOp>(const float *,
                                                              const float *,
                                                              float *,
                                                              size_t,
                                                              size_t,
                                                              size_t,
                                                              DivideOp,
                                                              cudaStream_t);
template void launch_simple_broadcast_kernel<double, DivideOp>(const double *,
                                                               const double *,
                                                               double *,
                                                               size_t,
                                                               size_t,
                                                               size_t,
                                                               DivideOp,
                                                               cudaStream_t);
template void launch_simple_broadcast_kernel<int32_t, DivideOp>(const int32_t *,
                                                                const int32_t *,
                                                                int32_t *,
                                                                size_t,
                                                                size_t,
                                                                size_t,
                                                                DivideOp,
                                                                cudaStream_t);
template void launch_simple_broadcast_kernel<int8_t, DivideOp>(const int8_t *,
                                                               const int8_t *,
                                                               int8_t *,
                                                               size_t,
                                                               size_t,
                                                               size_t,
                                                               DivideOp,
                                                               cudaStream_t);
template void launch_simple_broadcast_kernel<short, DivideOp>(const short *,
                                                              const short *,
                                                              short *,
                                                              size_t,
                                                              size_t,
                                                              size_t,
                                                              DivideOp,
                                                              cudaStream_t);
template void launch_simple_broadcast_kernel<long, DivideOp>(const long *,
                                                             const long *,
                                                             long *,
                                                             size_t,
                                                             size_t,
                                                             size_t,
                                                             DivideOp,
                                                             cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned char, DivideOp>(const unsigned char *,
                                                                      const unsigned char *,
                                                                      unsigned char *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      DivideOp,
                                                                      cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned short, DivideOp>(const unsigned short *,
                                                                       const unsigned short *,
                                                                       unsigned short *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       DivideOp,
                                                                       cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned int, DivideOp>(const unsigned int *,
                                                                     const unsigned int *,
                                                                     unsigned int *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     DivideOp,
                                                                     cudaStream_t);
template void launch_simple_broadcast_kernel<unsigned long, DivideOp>(const unsigned long *,
                                                                      const unsigned long *,
                                                                      unsigned long *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      DivideOp,
                                                                      cudaStream_t);
template void launch_simple_broadcast_kernel<bool, DivideOp>(const bool *,
                                                             const bool *,
                                                             bool *,
                                                             size_t,
                                                             size_t,
                                                             size_t,
                                                             DivideOp,
                                                             cudaStream_t);

// ============================================================================
// 类型转换内核的显式实例化
// ============================================================================
// 支持所有类型组合：11种类型 × 11种类型 = 121种组合
// 按照 DataType 枚举顺序排列：float, double, int8_t, int16_t, int32_t, int64_t,
//                              uint8_t, uint16_t, uint32_t, uint64_t, bool

// float -> 所有类型
template void launch_type_conversion_kernel<float, float>(const float *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, double>(const float *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, int8_t>(const float *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, int16_t>(const float *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, int32_t>(const float *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, int64_t>(const float *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, uint8_t>(const float *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, uint16_t>(const float *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, uint32_t>(const float *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, uint64_t>(const float *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<float, bool>(const float *, bool *, size_t, cudaStream_t);

// double -> 所有类型
template void launch_type_conversion_kernel<double, float>(const double *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, double>(const double *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, int8_t>(const double *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, int16_t>(const double *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, int32_t>(const double *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, int64_t>(const double *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, uint8_t>(const double *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, uint16_t>(const double *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, uint32_t>(const double *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, uint64_t>(const double *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<double, bool>(const double *, bool *, size_t, cudaStream_t);

// int8_t -> 所有类型
template void launch_type_conversion_kernel<int8_t, float>(const int8_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, double>(const int8_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, int8_t>(const int8_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, int16_t>(const int8_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, int32_t>(const int8_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, int64_t>(const int8_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, uint8_t>(const int8_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, uint16_t>(const int8_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, uint32_t>(const int8_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, uint64_t>(const int8_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int8_t, bool>(const int8_t *, bool *, size_t, cudaStream_t);

// int16_t -> 所有类型
template void launch_type_conversion_kernel<int16_t, float>(const int16_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, double>(const int16_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, int8_t>(const int16_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, int16_t>(const int16_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, int32_t>(const int16_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, int64_t>(const int16_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, uint8_t>(const int16_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, uint16_t>(const int16_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, uint32_t>(const int16_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, uint64_t>(const int16_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int16_t, bool>(const int16_t *, bool *, size_t, cudaStream_t);

// int32_t -> 所有类型
template void launch_type_conversion_kernel<int32_t, float>(const int32_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, double>(const int32_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, int8_t>(const int32_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, int16_t>(const int32_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, int32_t>(const int32_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, int64_t>(const int32_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, uint8_t>(const int32_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, uint16_t>(const int32_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, uint32_t>(const int32_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, uint64_t>(const int32_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int32_t, bool>(const int32_t *, bool *, size_t, cudaStream_t);

// int64_t -> 所有类型
template void launch_type_conversion_kernel<int64_t, float>(const int64_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, double>(const int64_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, int8_t>(const int64_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, int16_t>(const int64_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, int32_t>(const int64_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, int64_t>(const int64_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, uint8_t>(const int64_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, uint16_t>(const int64_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, uint32_t>(const int64_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, uint64_t>(const int64_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<int64_t, bool>(const int64_t *, bool *, size_t, cudaStream_t);

// uint8_t -> 所有类型
template void launch_type_conversion_kernel<uint8_t, float>(const uint8_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, double>(const uint8_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, int8_t>(const uint8_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, int16_t>(const uint8_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, int32_t>(const uint8_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, int64_t>(const uint8_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, uint8_t>(const uint8_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, uint16_t>(const uint8_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, uint32_t>(const uint8_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, uint64_t>(const uint8_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint8_t, bool>(const uint8_t *, bool *, size_t, cudaStream_t);

// uint16_t -> 所有类型
template void launch_type_conversion_kernel<uint16_t, float>(const uint16_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, double>(const uint16_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, int8_t>(const uint16_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, int16_t>(const uint16_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, int32_t>(const uint16_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, int64_t>(const uint16_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, uint8_t>(const uint16_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, uint16_t>(const uint16_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, uint32_t>(const uint16_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, uint64_t>(const uint16_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint16_t, bool>(const uint16_t *, bool *, size_t, cudaStream_t);

// uint32_t -> 所有类型
template void launch_type_conversion_kernel<uint32_t, float>(const uint32_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, double>(const uint32_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, int8_t>(const uint32_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, int16_t>(const uint32_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, int32_t>(const uint32_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, int64_t>(const uint32_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, uint8_t>(const uint32_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, uint16_t>(const uint32_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, uint32_t>(const uint32_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, uint64_t>(const uint32_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint32_t, bool>(const uint32_t *, bool *, size_t, cudaStream_t);

// uint64_t -> 所有类型
template void launch_type_conversion_kernel<uint64_t, float>(const uint64_t *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, double>(const uint64_t *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, int8_t>(const uint64_t *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, int16_t>(const uint64_t *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, int32_t>(const uint64_t *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, int64_t>(const uint64_t *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, uint8_t>(const uint64_t *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, uint16_t>(const uint64_t *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, uint32_t>(const uint64_t *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, uint64_t>(const uint64_t *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<uint64_t, bool>(const uint64_t *, bool *, size_t, cudaStream_t);

// bool -> 所有类型
template void launch_type_conversion_kernel<bool, float>(const bool *, float *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, double>(const bool *, double *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, int8_t>(const bool *, int8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, int16_t>(const bool *, int16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, int32_t>(const bool *, int32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, int64_t>(const bool *, int64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, uint8_t>(const bool *, uint8_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, uint16_t>(const bool *, uint16_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, uint32_t>(const bool *, uint32_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, uint64_t>(const bool *, uint64_t *, size_t, cudaStream_t);
template void launch_type_conversion_kernel<bool, bool>(const bool *, bool *, size_t, cudaStream_t);

// ============================================================================
// launch_index_put_kernel 模板实例化
// ============================================================================

template void launch_index_put_kernel<float>(float *, size_t, float, cudaStream_t);
template void launch_index_put_kernel<double>(double *, size_t, double, cudaStream_t);
template void launch_index_put_kernel<int32_t>(int32_t *, size_t, int32_t, cudaStream_t);
template void launch_index_put_kernel<int8_t>(int8_t *, size_t, int8_t, cudaStream_t);
template void launch_index_put_kernel<int16_t>(int16_t *, size_t, int16_t, cudaStream_t);
template void launch_index_put_kernel<int64_t>(int64_t *, size_t, int64_t, cudaStream_t);
template void launch_index_put_kernel<uint8_t>(uint8_t *, size_t, uint8_t, cudaStream_t);
template void launch_index_put_kernel<uint16_t>(uint16_t *, size_t, uint16_t, cudaStream_t);
template void launch_index_put_kernel<uint32_t>(uint32_t *, size_t, uint32_t, cudaStream_t);
template void launch_index_put_kernel<uint64_t>(uint64_t *, size_t, uint64_t, cudaStream_t);
template void launch_index_put_kernel<bool>(bool *, size_t, bool, cudaStream_t);

}  // namespace cuda
}  // namespace origin