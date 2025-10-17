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
// 向量化优化内核实现
// ============================================================================

/**
 * @brief 向量化二元运算内核实现
 * @details 使用float4/double4进行向量化操作，一次处理4个元素
 */
template <typename T, typename Op>
__global__ void vectorized_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, size_t n, Op op)
{
    // 计算向量化索引（一次处理4个元素）
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n)
    {
        // 使用float4进行向量化操作（仅对float类型）
        if (sizeof(T) == sizeof(float))
        {
            float4 a_vec = reinterpret_cast<const float4 *>(a)[idx / 4];
            float4 b_vec = reinterpret_cast<const float4 *>(b)[idx / 4];
            float4 c_vec =
                make_float4(op(a_vec.x, b_vec.x), op(a_vec.y, b_vec.y), op(a_vec.z, b_vec.z), op(a_vec.w, b_vec.w));
            reinterpret_cast<float4 *>(c)[idx / 4] = c_vec;
        }
        else
        {
            // 对于非float类型，回退到标量操作
            for (size_t i = 0; i < 4 && idx + i < n; ++i)
            {
                c[idx + i] = op(a[idx + i], b[idx + i]);
            }
        }
    }
    else
    {
        // 处理剩余的元素（不足4个）
        for (size_t i = idx; likely(i < n); ++i)
        {
            c[i] = op(a[i], b[i]);
        }
    }
}

/**
 * @brief 向量化一元运算内核实现
 */
template <typename T, typename Op>
__global__ void vectorized_unary_kernel(const T *__restrict__ a, T *__restrict__ c, size_t n, Op op)
{
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    if (idx + 3 < n)
    {
        if (sizeof(T) == sizeof(float))
        {
            float4 a_vec                           = reinterpret_cast<const float4 *>(a)[idx / 4];
            float4 c_vec                           = make_float4(op(a_vec.x), op(a_vec.y), op(a_vec.z), op(a_vec.w));
            reinterpret_cast<float4 *>(c)[idx / 4] = c_vec;
        }
        else
        {
            for (size_t i = 0; i < 4 && idx + i < n; ++i)
            {
                c[idx + i] = op(a[idx + i]);
            }
        }
    }
    else
    {
        for (size_t i = idx; likely(i < n); ++i)
        {
            c[i] = op(a[i]);
        }
    }
}

// ============================================================================
// 共享内存优化内核实现
// ============================================================================

/**
 * @brief 使用共享内存的二元运算内核实现
 * @details 将数据加载到共享内存中，减少全局内存访问次数
 */
template <typename T, typename Op>
__global__ void shared_memory_kernel(const T *__restrict__ a,
                                     const T *__restrict__ b,
                                     T *__restrict__ c,
                                     size_t n,
                                     Op op)
{
    // 声明共享内存
    extern __shared__ char shared_mem[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将数据加载到共享内存
    T *shared_a = reinterpret_cast<T *>(shared_mem);
    T *shared_b = reinterpret_cast<T *>(shared_mem + blockDim.x * sizeof(T));

    if (idx < n)
    {
        shared_a[tid] = a[idx];
        shared_b[tid] = b[idx];
    }

    // 同步所有线程，确保数据加载完成
    __syncthreads();

    // 从共享内存读取数据，执行操作
    if (idx < n)
    {
        c[idx] = op(shared_a[tid], shared_b[tid]);
    }
}

// ============================================================================
// 广播运算内核实现
// ============================================================================

/**
 * @brief 简单广播内核实现（一个操作数是标量）
 */
template <typename T, typename Op>
__global__ void scalar_broadcast_kernel(const T *__restrict__ a,
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
// 矩阵运算内核实现
// ============================================================================

/**
 * @brief 基础矩阵乘法内核实现
 */
template <typename T>
__global__ void matmul_kernel(const T *__restrict__ a, const T *__restrict__ b, T *__restrict__ c, int M, int N, int K)
{
    // 计算当前线程处理的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (likely(row < M && col < N))
    {
        T sum = 0;
        // 计算矩阵乘法的内积
        for (int k = 0; likely(k < K); ++k)
        {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

/**
 * @brief 分块矩阵乘法内核实现（使用共享内存优化）
 */
template <typename T>
__global__ void matmul_tiled_kernel(const T *__restrict__ a,
                                    const T *__restrict__ b,
                                    T *__restrict__ c,
                                    int M,
                                    int N,
                                    int K)
{
    const int TILE_SIZE = 16;

    // 声明共享内存
    __shared__ T tile_a[TILE_SIZE][TILE_SIZE];
    __shared__ T tile_b[TILE_SIZE][TILE_SIZE];

    // 计算当前线程处理的行和列
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    // 分块处理
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        // 加载tile到共享内存
        if (row < M && tile * TILE_SIZE + threadIdx.x < K)
        {
            tile_a[threadIdx.y][threadIdx.x] = a[row * K + tile * TILE_SIZE + threadIdx.x];
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }

        if (tile * TILE_SIZE + threadIdx.y < K && col < N)
        {
            tile_b[threadIdx.y][threadIdx.x] = b[(tile * TILE_SIZE + threadIdx.y) * N + col];
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }

        // 同步所有线程
        __syncthreads();

        // 计算tile内的乘积
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }

        // 同步所有线程
        __syncthreads();
    }

    // 写入结果
    if (row < M && col < N)
    {
        c[row * N + col] = sum;
    }
}

// ============================================================================
// 归约运算内核实现
// ============================================================================

/**
 * @brief 轴求和内核实现
 */
template <typename T>
__global__ void axis_sum_kernel(const T *__restrict__ input,
                                T *__restrict__ output,
                                const int *input_shape,
                                const int *output_shape,
                                int axis,
                                int ndims,
                                size_t input_elements,
                                size_t output_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < output_elements)
    {
        // 将输出索引转换为多维索引
        int output_indices[8];
        int temp = idx;
        for (int i = ndims - 2; i >= 0; --i)  // ndims-2因为去掉了axis维度
        {
            output_indices[i] = temp % output_shape[i];
            temp /= output_shape[i];
        }

        // 计算输入索引并求和
        T sum = 0;
        for (int axis_val = 0; axis_val < input_shape[axis]; ++axis_val)
        {
            int input_idx = 0;
            int stride    = 1;

            // 计算输入线性索引
            for (int i = ndims - 1; i >= 0; --i)
            {
                int dim_idx = (i == axis) ? axis_val : output_indices[(i < axis) ? i : i - 1];
                input_idx += dim_idx * stride;
                stride *= input_shape[i];
            }

            sum += input[input_idx];
        }

        output[idx] = sum;
    }
}

/**
 * @brief 全元素求和内核实现（使用树状归约）
 */
template <typename T>
__global__ void sum_all_kernel(const T *__restrict__ input, T *__restrict__ output, size_t n)
{
    extern __shared__ T shared_mem[];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    shared_mem[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();

    // 树状归约
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }

    // 第一个线程写入结果
    if (tid == 0)
    {
        output[blockIdx.x] = shared_mem[0];
    }
}

// ============================================================================
// 形状操作内核实现
// ============================================================================

/**
 * @brief 转置内核实现
 */
template <typename T>
__global__ void transpose_kernel(const T *__restrict__ input, T *__restrict__ output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        output[col * rows + row] = input[row * cols + col];
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

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA elementwise kernel launch failed: {}", cudaGetErrorString(err));
    }
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA unary kernel launch failed: {}", cudaGetErrorString(err));
    }
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

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA scalar kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 启动矩阵乘法内核
 */
template <typename T>
void launch_matmul_kernel(const T *a, const T *b, T *c, int M, int N, int K, cudaStream_t stream)
{
    // 对于矩阵乘法，使用2D网格
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // 根据矩阵大小选择内核
    if (M >= 64 && N >= 64 && K >= 64)
    {
        // 大矩阵使用分块内核
        matmul_tiled_kernel<T><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
    }
    else
    {
        // 小矩阵使用基础内核
        matmul_kernel<T><<<grid, block, 0, stream>>>(a, b, c, M, N, K);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA matmul kernel launch failed: {}", cudaGetErrorString(err));
    }
}

// ============================================================================
// 显式模板实例化
// ============================================================================

// 基础内核的显式实例化
template __global__ void elementwise_kernel<float, AddOp>(const float *, const float *, float *, size_t, AddOp);
template __global__ void elementwise_kernel<double, AddOp>(const double *, const double *, double *, size_t, AddOp);
template __global__ void elementwise_kernel<int32_t, AddOp>(const int32_t *, const int32_t *, int32_t *, size_t, AddOp);
template __global__ void elementwise_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, AddOp);

template __global__ void elementwise_kernel<float, SubtractOp>(const float *,
                                                               const float *,
                                                               float *,
                                                               size_t,
                                                               SubtractOp);
template __global__ void elementwise_kernel<double, SubtractOp>(const double *,
                                                                const double *,
                                                                double *,
                                                                size_t,
                                                                SubtractOp);
template __global__ void elementwise_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                 const int32_t *,
                                                                 int32_t *,
                                                                 size_t,
                                                                 SubtractOp);
template __global__ void elementwise_kernel<int8_t, SubtractOp>(const int8_t *,
                                                                const int8_t *,
                                                                int8_t *,
                                                                size_t,
                                                                SubtractOp);

template __global__ void elementwise_kernel<float, MultiplyOp>(const float *,
                                                               const float *,
                                                               float *,
                                                               size_t,
                                                               MultiplyOp);
template __global__ void elementwise_kernel<double, MultiplyOp>(const double *,
                                                                const double *,
                                                                double *,
                                                                size_t,
                                                                MultiplyOp);
template __global__ void elementwise_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                 const int32_t *,
                                                                 int32_t *,
                                                                 size_t,
                                                                 MultiplyOp);
template __global__ void elementwise_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                                const int8_t *,
                                                                int8_t *,
                                                                size_t,
                                                                MultiplyOp);

template __global__ void elementwise_kernel<float, DivideOp>(const float *, const float *, float *, size_t, DivideOp);
template __global__ void elementwise_kernel<double, DivideOp>(const double *,
                                                              const double *,
                                                              double *,
                                                              size_t,
                                                              DivideOp);
template __global__ void elementwise_kernel<int32_t, DivideOp>(const int32_t *,
                                                               const int32_t *,
                                                               int32_t *,
                                                               size_t,
                                                               DivideOp);
template __global__ void elementwise_kernel<int8_t, DivideOp>(const int8_t *,
                                                              const int8_t *,
                                                              int8_t *,
                                                              size_t,
                                                              DivideOp);

// 一元内核的显式实例化
template __global__ void unary_kernel<float, ExpOp>(const float *, float *, size_t, ExpOp);
template __global__ void unary_kernel<double, ExpOp>(const double *, double *, size_t, ExpOp);
template __global__ void unary_kernel<float, LogOp>(const float *, float *, size_t, LogOp);
template __global__ void unary_kernel<double, LogOp>(const double *, double *, size_t, LogOp);
template __global__ void unary_kernel<float, SqrtOp>(const float *, float *, size_t, SqrtOp);
template __global__ void unary_kernel<double, SqrtOp>(const double *, double *, size_t, SqrtOp);
template __global__ void unary_kernel<float, SquareOp>(const float *, float *, size_t, SquareOp);
template __global__ void unary_kernel<double, SquareOp>(const double *, double *, size_t, SquareOp);
template __global__ void unary_kernel<int32_t, SquareOp>(const int32_t *, int32_t *, size_t, SquareOp);
template __global__ void unary_kernel<int8_t, SquareOp>(const int8_t *, int8_t *, size_t, SquareOp);
template __global__ void unary_kernel<short, SquareOp>(const short *, short *, size_t, SquareOp);
template __global__ void unary_kernel<long, SquareOp>(const long *, long *, size_t, SquareOp);
template __global__ void unary_kernel<unsigned char, SquareOp>(const unsigned char *,
                                                               unsigned char *,
                                                               size_t,
                                                               SquareOp);
template __global__ void unary_kernel<unsigned short, SquareOp>(const unsigned short *,
                                                                unsigned short *,
                                                                size_t,
                                                                SquareOp);
template __global__ void unary_kernel<unsigned int, SquareOp>(const unsigned int *, unsigned int *, size_t, SquareOp);
template __global__ void unary_kernel<unsigned long, SquareOp>(const unsigned long *,
                                                               unsigned long *,
                                                               size_t,
                                                               SquareOp);
template __global__ void unary_kernel<bool, SquareOp>(const bool *, bool *, size_t, SquareOp);
template __global__ void unary_kernel<float, NegOp>(const float *, float *, size_t, NegOp);
template __global__ void unary_kernel<double, NegOp>(const double *, double *, size_t, NegOp);
template __global__ void unary_kernel<int32_t, NegOp>(const int32_t *, int32_t *, size_t, NegOp);
template __global__ void unary_kernel<int8_t, NegOp>(const int8_t *, int8_t *, size_t, NegOp);
template __global__ void unary_kernel<short, NegOp>(const short *, short *, size_t, NegOp);
template __global__ void unary_kernel<long, NegOp>(const long *, long *, size_t, NegOp);
template __global__ void unary_kernel<unsigned char, NegOp>(const unsigned char *, unsigned char *, size_t, NegOp);
template __global__ void unary_kernel<unsigned short, NegOp>(const unsigned short *, unsigned short *, size_t, NegOp);
template __global__ void unary_kernel<unsigned int, NegOp>(const unsigned int *, unsigned int *, size_t, NegOp);
template __global__ void unary_kernel<unsigned long, NegOp>(const unsigned long *, unsigned long *, size_t, NegOp);
template __global__ void unary_kernel<bool, NegOp>(const bool *, bool *, size_t, NegOp);

// 标量内核的显式实例化
template __global__ void scalar_kernel<float, AddOp>(const float *, float, float *, size_t, AddOp);
template __global__ void scalar_kernel<double, AddOp>(const double *, double, double *, size_t, AddOp);
template __global__ void scalar_kernel<int32_t, AddOp>(const int32_t *, int32_t, int32_t *, size_t, AddOp);
template __global__ void scalar_kernel<int8_t, AddOp>(const int8_t *, int8_t, int8_t *, size_t, AddOp);

template __global__ void scalar_kernel<float, MultiplyOp>(const float *, float, float *, size_t, MultiplyOp);
template __global__ void scalar_kernel<double, MultiplyOp>(const double *, double, double *, size_t, MultiplyOp);
template __global__ void scalar_kernel<int32_t, MultiplyOp>(const int32_t *, int32_t, int32_t *, size_t, MultiplyOp);
template __global__ void scalar_kernel<int8_t, MultiplyOp>(const int8_t *, int8_t, int8_t *, size_t, MultiplyOp);

// 矩阵乘法内核的显式实例化
template __global__ void matmul_kernel<float>(const float *, const float *, float *, int, int, int);
template __global__ void matmul_kernel<double>(const double *, const double *, double *, int, int, int);
template __global__ void matmul_tiled_kernel<float>(const float *, const float *, float *, int, int, int);
template __global__ void matmul_tiled_kernel<double>(const double *, const double *, double *, int, int, int);

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

template void launch_unary_kernel<float, ExpOp>(const float *, float *, size_t, ExpOp, cudaStream_t);
template void launch_unary_kernel<double, ExpOp>(const double *, double *, size_t, ExpOp, cudaStream_t);
template void launch_unary_kernel<float, LogOp>(const float *, float *, size_t, LogOp, cudaStream_t);
template void launch_unary_kernel<double, LogOp>(const double *, double *, size_t, LogOp, cudaStream_t);
template void launch_unary_kernel<float, SqrtOp>(const float *, float *, size_t, SqrtOp, cudaStream_t);
template void launch_unary_kernel<double, SqrtOp>(const double *, double *, size_t, SqrtOp, cudaStream_t);
template void launch_unary_kernel<float, SquareOp>(const float *, float *, size_t, SquareOp, cudaStream_t);
template void launch_unary_kernel<double, SquareOp>(const double *, double *, size_t, SquareOp, cudaStream_t);
template void launch_unary_kernel<int32_t, SquareOp>(const int32_t *, int32_t *, size_t, SquareOp, cudaStream_t);
template void launch_unary_kernel<int8_t, SquareOp>(const int8_t *, int8_t *, size_t, SquareOp, cudaStream_t);
template void launch_unary_kernel<float, NegOp>(const float *, float *, size_t, NegOp, cudaStream_t);
template void launch_unary_kernel<double, NegOp>(const double *, double *, size_t, NegOp, cudaStream_t);
template void launch_unary_kernel<int32_t, NegOp>(const int32_t *, int32_t *, size_t, NegOp, cudaStream_t);
template void launch_unary_kernel<int8_t, NegOp>(const int8_t *, int8_t *, size_t, NegOp, cudaStream_t);

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

template void launch_matmul_kernel<float>(const float *, const float *, float *, int, int, int, cudaStream_t);
template void launch_matmul_kernel<double>(const double *, const double *, double *, int, int, int, cudaStream_t);

// 向量化内核的显式实例化
template __global__ void vectorized_kernel<float, AddOp>(const float *, const float *, float *, size_t, AddOp);
template __global__ void vectorized_kernel<double, AddOp>(const double *, const double *, double *, size_t, AddOp);
template __global__ void vectorized_kernel<int32_t, AddOp>(const int32_t *, const int32_t *, int32_t *, size_t, AddOp);
template __global__ void vectorized_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, AddOp);

template __global__ void vectorized_kernel<float, SubtractOp>(const float *,
                                                              const float *,
                                                              float *,
                                                              size_t,
                                                              SubtractOp);
template __global__ void vectorized_kernel<double, SubtractOp>(const double *,
                                                               const double *,
                                                               double *,
                                                               size_t,
                                                               SubtractOp);
template __global__ void vectorized_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                const int32_t *,
                                                                int32_t *,
                                                                size_t,
                                                                SubtractOp);
template __global__ void vectorized_kernel<int8_t, SubtractOp>(const int8_t *,
                                                               const int8_t *,
                                                               int8_t *,
                                                               size_t,
                                                               SubtractOp);

template __global__ void vectorized_kernel<float, MultiplyOp>(const float *,
                                                              const float *,
                                                              float *,
                                                              size_t,
                                                              MultiplyOp);
template __global__ void vectorized_kernel<double, MultiplyOp>(const double *,
                                                               const double *,
                                                               double *,
                                                               size_t,
                                                               MultiplyOp);
template __global__ void vectorized_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                const int32_t *,
                                                                int32_t *,
                                                                size_t,
                                                                MultiplyOp);
template __global__ void vectorized_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                               const int8_t *,
                                                               int8_t *,
                                                               size_t,
                                                               MultiplyOp);

template __global__ void vectorized_kernel<float, DivideOp>(const float *, const float *, float *, size_t, DivideOp);
template __global__ void vectorized_kernel<double, DivideOp>(const double *,
                                                             const double *,
                                                             double *,
                                                             size_t,
                                                             DivideOp);
template __global__ void vectorized_kernel<int32_t, DivideOp>(const int32_t *,
                                                              const int32_t *,
                                                              int32_t *,
                                                              size_t,
                                                              DivideOp);
template __global__ void vectorized_kernel<int8_t, DivideOp>(const int8_t *,
                                                             const int8_t *,
                                                             int8_t *,
                                                             size_t,
                                                             DivideOp);

// 向量化一元内核的显式实例化
template __global__ void vectorized_unary_kernel<float, ExpOp>(const float *, float *, size_t, ExpOp);
template __global__ void vectorized_unary_kernel<double, ExpOp>(const double *, double *, size_t, ExpOp);
template __global__ void vectorized_unary_kernel<float, LogOp>(const float *, float *, size_t, LogOp);
template __global__ void vectorized_unary_kernel<double, LogOp>(const double *, double *, size_t, LogOp);
template __global__ void vectorized_unary_kernel<float, SqrtOp>(const float *, float *, size_t, SqrtOp);
template __global__ void vectorized_unary_kernel<double, SqrtOp>(const double *, double *, size_t, SqrtOp);
template __global__ void vectorized_unary_kernel<float, SquareOp>(const float *, float *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<double, SquareOp>(const double *, double *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<int32_t, SquareOp>(const int32_t *, int32_t *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<int8_t, SquareOp>(const int8_t *, int8_t *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<short, SquareOp>(const short *, short *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<long, SquareOp>(const long *, long *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<unsigned char, SquareOp>(const unsigned char *,
                                                                          unsigned char *,
                                                                          size_t,
                                                                          SquareOp);
template __global__ void vectorized_unary_kernel<unsigned short, SquareOp>(const unsigned short *,
                                                                           unsigned short *,
                                                                           size_t,
                                                                           SquareOp);
template __global__ void vectorized_unary_kernel<unsigned int, SquareOp>(const unsigned int *,
                                                                         unsigned int *,
                                                                         size_t,
                                                                         SquareOp);
template __global__ void vectorized_unary_kernel<unsigned long, SquareOp>(const unsigned long *,
                                                                          unsigned long *,
                                                                          size_t,
                                                                          SquareOp);
template __global__ void vectorized_unary_kernel<bool, SquareOp>(const bool *, bool *, size_t, SquareOp);
template __global__ void vectorized_unary_kernel<float, NegOp>(const float *, float *, size_t, NegOp);
template __global__ void vectorized_unary_kernel<double, NegOp>(const double *, double *, size_t, NegOp);
template __global__ void vectorized_unary_kernel<int32_t, NegOp>(const int32_t *, int32_t *, size_t, NegOp);
template __global__ void vectorized_unary_kernel<int8_t, NegOp>(const int8_t *, int8_t *, size_t, NegOp);
template __global__ void vectorized_unary_kernel<short, NegOp>(const short *, short *, size_t, NegOp);
template __global__ void vectorized_unary_kernel<long, NegOp>(const long *, long *, size_t, NegOp);
template __global__ void vectorized_unary_kernel<unsigned char, NegOp>(const unsigned char *,
                                                                       unsigned char *,
                                                                       size_t,
                                                                       NegOp);
template __global__ void vectorized_unary_kernel<unsigned short, NegOp>(const unsigned short *,
                                                                        unsigned short *,
                                                                        size_t,
                                                                        NegOp);
template __global__ void vectorized_unary_kernel<unsigned int, NegOp>(const unsigned int *,
                                                                      unsigned int *,
                                                                      size_t,
                                                                      NegOp);
template __global__ void vectorized_unary_kernel<unsigned long, NegOp>(const unsigned long *,
                                                                       unsigned long *,
                                                                       size_t,
                                                                       NegOp);
template __global__ void vectorized_unary_kernel<bool, NegOp>(const bool *, bool *, size_t, NegOp);

// 共享内存内核的显式实例化
template __global__ void shared_memory_kernel<float, AddOp>(const float *, const float *, float *, size_t, AddOp);
template __global__ void shared_memory_kernel<double, AddOp>(const double *, const double *, double *, size_t, AddOp);
template __global__ void shared_memory_kernel<int32_t, AddOp>(const int32_t *,
                                                              const int32_t *,
                                                              int32_t *,
                                                              size_t,
                                                              AddOp);
template __global__ void shared_memory_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, AddOp);

template __global__ void shared_memory_kernel<float, SubtractOp>(const float *,
                                                                 const float *,
                                                                 float *,
                                                                 size_t,
                                                                 SubtractOp);
template __global__ void shared_memory_kernel<double, SubtractOp>(const double *,
                                                                  const double *,
                                                                  double *,
                                                                  size_t,
                                                                  SubtractOp);
template __global__ void shared_memory_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                   const int32_t *,
                                                                   int32_t *,
                                                                   size_t,
                                                                   SubtractOp);
template __global__ void shared_memory_kernel<int8_t, SubtractOp>(const int8_t *,
                                                                  const int8_t *,
                                                                  int8_t *,
                                                                  size_t,
                                                                  SubtractOp);

template __global__ void shared_memory_kernel<float, MultiplyOp>(const float *,
                                                                 const float *,
                                                                 float *,
                                                                 size_t,
                                                                 MultiplyOp);
template __global__ void shared_memory_kernel<double, MultiplyOp>(const double *,
                                                                  const double *,
                                                                  double *,
                                                                  size_t,
                                                                  MultiplyOp);
template __global__ void shared_memory_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                   const int32_t *,
                                                                   int32_t *,
                                                                   size_t,
                                                                   MultiplyOp);
template __global__ void shared_memory_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                                  const int8_t *,
                                                                  int8_t *,
                                                                  size_t,
                                                                  MultiplyOp);

template __global__ void shared_memory_kernel<float, DivideOp>(const float *, const float *, float *, size_t, DivideOp);
template __global__ void shared_memory_kernel<double, DivideOp>(const double *,
                                                                const double *,
                                                                double *,
                                                                size_t,
                                                                DivideOp);
template __global__ void shared_memory_kernel<int32_t, DivideOp>(const int32_t *,
                                                                 const int32_t *,
                                                                 int32_t *,
                                                                 size_t,
                                                                 DivideOp);
template __global__ void shared_memory_kernel<int8_t, DivideOp>(const int8_t *,
                                                                const int8_t *,
                                                                int8_t *,
                                                                size_t,
                                                                DivideOp);

// 标量广播内核的显式实例化
template __global__ void
scalar_broadcast_kernel<float, AddOp>(const float *, const float *, float *, size_t, size_t, size_t, AddOp);
template __global__ void
scalar_broadcast_kernel<double, AddOp>(const double *, const double *, double *, size_t, size_t, size_t, AddOp);
template __global__ void
scalar_broadcast_kernel<int32_t, AddOp>(const int32_t *, const int32_t *, int32_t *, size_t, size_t, size_t, AddOp);
template __global__ void
scalar_broadcast_kernel<int8_t, AddOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, AddOp);

template __global__ void
scalar_broadcast_kernel<float, SubtractOp>(const float *, const float *, float *, size_t, size_t, size_t, SubtractOp);
template __global__ void scalar_broadcast_kernel<double, SubtractOp>(const double *,
                                                                     const double *,
                                                                     double *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     SubtractOp);
template __global__ void scalar_broadcast_kernel<int32_t, SubtractOp>(const int32_t *,
                                                                      const int32_t *,
                                                                      int32_t *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      SubtractOp);
template __global__ void scalar_broadcast_kernel<int8_t, SubtractOp>(const int8_t *,
                                                                     const int8_t *,
                                                                     int8_t *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     SubtractOp);

template __global__ void
scalar_broadcast_kernel<float, MultiplyOp>(const float *, const float *, float *, size_t, size_t, size_t, MultiplyOp);
template __global__ void scalar_broadcast_kernel<double, MultiplyOp>(const double *,
                                                                     const double *,
                                                                     double *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     MultiplyOp);
template __global__ void scalar_broadcast_kernel<int32_t, MultiplyOp>(const int32_t *,
                                                                      const int32_t *,
                                                                      int32_t *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      MultiplyOp);
template __global__ void scalar_broadcast_kernel<int8_t, MultiplyOp>(const int8_t *,
                                                                     const int8_t *,
                                                                     int8_t *,
                                                                     size_t,
                                                                     size_t,
                                                                     size_t,
                                                                     MultiplyOp);

template __global__ void
scalar_broadcast_kernel<float, DivideOp>(const float *, const float *, float *, size_t, size_t, size_t, DivideOp);
template __global__ void
scalar_broadcast_kernel<double, DivideOp>(const double *, const double *, double *, size_t, size_t, size_t, DivideOp);
template __global__ void scalar_broadcast_kernel<int32_t, DivideOp>(const int32_t *,
                                                                    const int32_t *,
                                                                    int32_t *,
                                                                    size_t,
                                                                    size_t,
                                                                    size_t,
                                                                    DivideOp);
template __global__ void
scalar_broadcast_kernel<int8_t, DivideOp>(const int8_t *, const int8_t *, int8_t *, size_t, size_t, size_t, DivideOp);

// 添加缺失的数据类型实例化

// 标量广播内核的缺失实例化
template __global__ void
scalar_broadcast_kernel<short, AddOp>(const short *, const short *, short *, size_t, size_t, size_t, AddOp);
template __global__ void
scalar_broadcast_kernel<long, AddOp>(const long *, const long *, long *, size_t, size_t, size_t, AddOp);
template __global__ void scalar_broadcast_kernel<unsigned char, AddOp>(const unsigned char *,
                                                                       const unsigned char *,
                                                                       unsigned char *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       AddOp);
template __global__ void scalar_broadcast_kernel<unsigned short, AddOp>(const unsigned short *,
                                                                        const unsigned short *,
                                                                        unsigned short *,
                                                                        size_t,
                                                                        size_t,
                                                                        size_t,
                                                                        AddOp);
template __global__ void scalar_broadcast_kernel<unsigned int, AddOp>(const unsigned int *,
                                                                      const unsigned int *,
                                                                      unsigned int *,
                                                                      size_t,
                                                                      size_t,
                                                                      size_t,
                                                                      AddOp);
template __global__ void scalar_broadcast_kernel<unsigned long, AddOp>(const unsigned long *,
                                                                       const unsigned long *,
                                                                       unsigned long *,
                                                                       size_t,
                                                                       size_t,
                                                                       size_t,
                                                                       AddOp);
template __global__ void
scalar_broadcast_kernel<bool, AddOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, AddOp);

template __global__ void
scalar_broadcast_kernel<short, SubtractOp>(const short *, const short *, short *, size_t, size_t, size_t, SubtractOp);
template __global__ void
scalar_broadcast_kernel<long, SubtractOp>(const long *, const long *, long *, size_t, size_t, size_t, SubtractOp);
template __global__ void scalar_broadcast_kernel<unsigned char, SubtractOp>(const unsigned char *,
                                                                            const unsigned char *,
                                                                            unsigned char *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            SubtractOp);
template __global__ void scalar_broadcast_kernel<unsigned short, SubtractOp>(const unsigned short *,
                                                                             const unsigned short *,
                                                                             unsigned short *,
                                                                             size_t,
                                                                             size_t,
                                                                             size_t,
                                                                             SubtractOp);
template __global__ void scalar_broadcast_kernel<unsigned int, SubtractOp>(const unsigned int *,
                                                                           const unsigned int *,
                                                                           unsigned int *,
                                                                           size_t,
                                                                           size_t,
                                                                           size_t,
                                                                           SubtractOp);
template __global__ void scalar_broadcast_kernel<unsigned long, SubtractOp>(const unsigned long *,
                                                                            const unsigned long *,
                                                                            unsigned long *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            SubtractOp);
template __global__ void
scalar_broadcast_kernel<bool, SubtractOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, SubtractOp);

template __global__ void
scalar_broadcast_kernel<short, MultiplyOp>(const short *, const short *, short *, size_t, size_t, size_t, MultiplyOp);
template __global__ void
scalar_broadcast_kernel<long, MultiplyOp>(const long *, const long *, long *, size_t, size_t, size_t, MultiplyOp);
template __global__ void scalar_broadcast_kernel<unsigned char, MultiplyOp>(const unsigned char *,
                                                                            const unsigned char *,
                                                                            unsigned char *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            MultiplyOp);
template __global__ void scalar_broadcast_kernel<unsigned short, MultiplyOp>(const unsigned short *,
                                                                             const unsigned short *,
                                                                             unsigned short *,
                                                                             size_t,
                                                                             size_t,
                                                                             size_t,
                                                                             MultiplyOp);
template __global__ void scalar_broadcast_kernel<unsigned int, MultiplyOp>(const unsigned int *,
                                                                           const unsigned int *,
                                                                           unsigned int *,
                                                                           size_t,
                                                                           size_t,
                                                                           size_t,
                                                                           MultiplyOp);
template __global__ void scalar_broadcast_kernel<unsigned long, MultiplyOp>(const unsigned long *,
                                                                            const unsigned long *,
                                                                            unsigned long *,
                                                                            size_t,
                                                                            size_t,
                                                                            size_t,
                                                                            MultiplyOp);
template __global__ void
scalar_broadcast_kernel<bool, MultiplyOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, MultiplyOp);

template __global__ void
scalar_broadcast_kernel<short, DivideOp>(const short *, const short *, short *, size_t, size_t, size_t, DivideOp);
template __global__ void
scalar_broadcast_kernel<long, DivideOp>(const long *, const long *, long *, size_t, size_t, size_t, DivideOp);
template __global__ void scalar_broadcast_kernel<unsigned char, DivideOp>(const unsigned char *,
                                                                          const unsigned char *,
                                                                          unsigned char *,
                                                                          size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          DivideOp);
template __global__ void scalar_broadcast_kernel<unsigned short, DivideOp>(const unsigned short *,
                                                                           const unsigned short *,
                                                                           unsigned short *,
                                                                           size_t,
                                                                           size_t,
                                                                           size_t,
                                                                           DivideOp);
template __global__ void scalar_broadcast_kernel<unsigned int, DivideOp>(const unsigned int *,
                                                                         const unsigned int *,
                                                                         unsigned int *,
                                                                         size_t,
                                                                         size_t,
                                                                         size_t,
                                                                         DivideOp);
template __global__ void scalar_broadcast_kernel<unsigned long, DivideOp>(const unsigned long *,
                                                                          const unsigned long *,
                                                                          unsigned long *,
                                                                          size_t,
                                                                          size_t,
                                                                          size_t,
                                                                          DivideOp);
template __global__ void
scalar_broadcast_kernel<bool, DivideOp>(const bool *, const bool *, bool *, size_t, size_t, size_t, DivideOp);

}  // namespace cuda
}  // namespace origin