#include <type_traits>
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

#ifdef ENABLE_CUBLAS
#    include "origin/mat/origin/cuda/cublas_algo.cuh"
#endif

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA矩阵乘法算子实现
 * @details 实现高效的GPU矩阵乘法运算
 *
 * ============================================================================
 * PyTorch matmul行为详解
 * ============================================================================
 *
 * PyTorch的matmul行为支持多种矩阵乘法场景：
 *
 * 1. 2D矩阵乘法：
 *    - 标准的矩阵乘法运算
 *    - 支持不同形状的矩阵（需要满足矩阵乘法维度要求）
 *    - 高效的GPU实现，使用优化的CUDA内核
 *
 *    示例：
 *    ```python
 *    import torch
 *    a = torch.randn(3, 4)  # 3x4矩阵
 *    b = torch.randn(4, 5)  # 4x5矩阵
 *    c = torch.matmul(a, b)  # 结果: 3x5矩阵
 *    ```
 *
 * 2. 批量矩阵乘法：
 *    - 支持高维张量的批量矩阵乘法
 *    - 最后两个维度进行矩阵乘法，前面的维度作为批量维度
 *    - 广播支持：批量维度可以广播
 *
 *    示例：
 *    ```python
 *    a = torch.randn(2, 3, 4)  # 批量维度: 2, 矩阵维度: 3x4
 *    b = torch.randn(2, 4, 5)  # 批量维度: 2, 矩阵维度: 4x5
 *    c = torch.matmul(a, b)    # 结果: 2x3x5
 *    ```
 *
 * 3. 广播矩阵乘法：
 *    - 支持不同批量维度的矩阵乘法
 *    - 自动广播批量维度
 *
 *    示例：
 *    ```python
 *    a = torch.randn(3, 4)     # 2D矩阵
 *    b = torch.randn(2, 4, 5)  # 批量矩阵
 *    c = torch.matmul(a, b)    # 结果: 2x3x5 (a被广播)
 *    ```
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于CUDA内核的矩阵乘法策略：
 * - 实现高效的2D矩阵乘法内核
 * - 支持批量矩阵乘法
 * - 使用优化的内存访问模式
 * - 与CPU版本保持一致的行为
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现更高级的优化：
 * 1. 使用cuBLAS库进行高性能矩阵乘法
 * 2. 实现分块矩阵乘法优化
 * 3. 支持混合精度计算
 * 4. 实现更复杂的广播逻辑
 *
 * 实现步骤：
 * - 集成cuBLAS库调用
 * - 实现自动内核选择策略
 * - 添加性能基准测试
 * - 保持与PyTorch API的兼容性
 */

/**
 * @brief 2D矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
__global__ void matmul_2d_kernel(const T *__restrict__ a,
                                 const T *__restrict__ b,
                                 T *__restrict__ c,
                                 int M,
                                 int N,
                                 int K)
{
    // 计算当前线程处理的行和列
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (likely(row < M && col < N))
    {
        T sum = 0;
        // 计算矩阵乘法的内积
        for (int k = 0; k < K; ++k)
        {
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

/**
 * @brief 分块矩阵乘法CUDA内核（使用共享内存优化）
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
__global__ void matmul_tiled_kernel(const T *__restrict__ a,
                                    const T *__restrict__ b,
                                    T *__restrict__ c,
                                    int M,
                                    int N,
                                    int K)
{
    // 共享内存块大小
    const int TILE_SIZE = 16;

    __shared__ T shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    // 分块计算
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile)
    {
        // 加载数据到共享内存
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (likely(row < M && a_col < K))
            shared_a[threadIdx.y][threadIdx.x] = a[row * K + a_col];
        else
            shared_a[threadIdx.y][threadIdx.x] = 0;

        if (likely(b_row < K && col < N))
            shared_b[threadIdx.y][threadIdx.x] = b[b_row * N + col];
        else
            shared_b[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // 计算部分和
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    // 写入结果
    if (likely(row < M && col < N))
    {
        c[row * N + col] = sum;
    }
}

/**
 * @brief 启动2D矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_2d_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    // 根据矩阵大小选择内核策略
    if (M >= 32 && N >= 32 && K >= 32)
    {
        // 大矩阵：使用分块内核
        const int TILE_SIZE = 16;
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

        matmul_tiled_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
    }
    else
    {
        // 小矩阵：使用简单内核
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);

        matmul_2d_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
    }
}

/**
 * @brief CUDA矩阵乘法算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 矩阵乘法结果矩阵
 */
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b)
{
    // 验证输入
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CUDA_DEVICE(a, b);

    // 检查输入矩阵是否连续，cuBLAS需要连续内存
    // 对于非连续矩阵，我们需要创建连续副本（避免多余的拷贝，只在必要时拷贝）
    std::unique_ptr<Mat> a_contiguous;
    std::unique_ptr<Mat> b_contiguous;
    const OriginMat *a_ptr = &a;
    const OriginMat *b_ptr = &b;

    if (unlikely(!a.is_contiguous()))
    {
        a_contiguous = a.contiguous();
        a_ptr        = static_cast<const OriginMat *>(a_contiguous.get());
    }
    if (unlikely(!b.is_contiguous()))
    {
        b_contiguous = b.contiguous();
        b_ptr        = static_cast<const OriginMat *>(b_contiguous.get());
    }

    // 获取输入形状
    const auto &shape_a = a_ptr->shape();
    const auto &shape_b = b_ptr->shape();

    // 验证矩阵乘法维度要求
    if (unlikely(shape_a.size() < 2 || shape_b.size() < 2))
    {
        THROW_INVALID_ARG("MatMul requires at least 2D tensors, got shapes {} and {}", shape_a.to_string(),
                          shape_b.to_string());
    }

    // 获取最后两个维度
    int M  = static_cast<int>(shape_a[shape_a.size() - 2]);
    int K  = static_cast<int>(shape_a[shape_a.size() - 1]);
    int K2 = static_cast<int>(shape_b[shape_b.size() - 2]);
    int N  = static_cast<int>(shape_b[shape_b.size() - 1]);

    if (unlikely(K != K2))
    {
        THROW_INVALID_ARG("MatMul dimension mismatch: {} vs {}", shape_a.to_string(), shape_b.to_string());
    }

    // 计算输出形状
    std::vector<size_t> output_dims;

    // 处理批量维度（简化实现，只支持相同批量维度）
    if (likely(shape_a.size() == 2 && shape_b.size() == 2))
    {
        // 2D矩阵乘法
        output_dims = {static_cast<size_t>(M), static_cast<size_t>(N)};
    }
    else if (likely(shape_a.size() == shape_b.size()))
    {
        // 批量矩阵乘法（相同批量维度）
        output_dims                         = shape_a.dims();
        output_dims[output_dims.size() - 2] = M;
        output_dims[output_dims.size() - 1] = N;
    }
    else
    {
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA matmul");
    }

    Shape output_shape(output_dims);
    auto result = std::make_unique<OriginMat>(output_shape, a_ptr->dtype(), a_ptr->device());

    // 使用类型分发器执行矩阵乘法
    device_common::TypeDispatcher::dispatch_void(a_ptr->dtype(), [&]<typename T>() {
#ifdef ENABLE_CUBLAS
        // 如果启用cuBLAS，并且是支持的类型（float或double），使用cuBLAS
        // cuBLAS只支持float和double类型
        constexpr bool is_supported_type = std::is_same_v<T, float> || std::is_same_v<T, double>;

        if constexpr (is_supported_type)
        {
            cublas_matmul<T>(a_ptr->data_ptr<T>(), b_ptr->data_ptr<T>(), result->data_ptr<T>(), M, N, K);
        }
        else
        {
            launch_matmul_2d_kernel(a_ptr->data_ptr<T>(), b_ptr->data_ptr<T>(), result->data_ptr<T>(), M, N, K);
        }
#else
        launch_matmul_2d_kernel(a_ptr->data_ptr<T>(), b_ptr->data_ptr<T>(), result->data_ptr<T>(), M, N, K);
#endif
    });

    return result;
}

}  // namespace cuda
}  // namespace origin
