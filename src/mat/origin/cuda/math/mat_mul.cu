#include <type_traits>
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/env_config.h"
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
 * @details 实现高效的GPU矩阵乘法运算，包含多个优化版本
 *
 * Version 0: 朴素实现（Baseline）
 * 每个线程计算一个输出元素，直接从全局内存读取数据，作为其它优化版本的性能基线。
 *
 * Version 1: 共享内存分块（Shared Memory Tiling）
 * 使用共享内存缓存数据块，减少对全局内存的重复访问。
 *
 * Version 2: 优化共享内存访问（Bank Conflict Avoidance）
 * 使用 padding 避免共享内存 bank 冲突，提高共享内存带宽利用率。
 *
 * Version 3: 寄存器分块（Register Tiling）
 * 每个线程计算多个输出元素，利用寄存器保存中间结果，减少共享内存访问。
 *
 * Version 4: 向量化内存访问（Vectorized Memory Access）
 * 使用 float4 / float2 等向量类型进行加载，提高全局内存带宽利用率。
 *
 * Version 5: 双缓冲（Double Buffering）
 * 重叠计算和内存加载，在计算当前 tile 时预取下一 tile 数据，隐藏全局内存访问延迟。
 *
 * Version 6: 循环展开（Loop Unrolling）
 * 手动展开内层循环，减少循环控制开销并提高指令级并行度。
 *
 * Version 7: Warptiling（Warp-level Tiling）
 * 在 block 和 thread 之间增加 warp 层，每个 warp 负责一个子块，更好地利用 warp 局部性并减少 shared memory bank 冲突。
 *
 */

/**
 * @brief 矩阵乘法版本枚举
 */
enum class MatMulVersion
{
    V0_NAIVE              = 0,     // 朴素实现
    V1_TILED              = 1,     // 共享内存分块
    V2_BANK_CONFLICT_FREE = 2,     // 避免bank冲突
    V3_REGISTER_TILING    = 3,     // 寄存器分块
    V4_VECTORIZED         = 4,     // 向量化（仅float）
    V5_DOUBLE_BUFFERING   = 5,     // 双缓冲
    V6_UNROLLED           = 6,     // 循环展开
    V7_WARPTILING         = 7,     // Warptiling
    V_AUTO                = 6666,  // 自动选择最优版本
};

// ============================================================================
// Version 0: 朴素实现（Baseline）
// ============================================================================

/**
 * @brief Version 0: 朴素矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 朴素矩阵乘法的基线实现：
 * 每个线程计算一个输出元素 C[row, col]。
 * 不使用共享内存，直接从全局内存加载 A、B。
 * 计算公式为 C[row][col] = sum(A[row][k] * B[k][col]) for k in [0, K)。
 *
 * 性能特征：
 * A 为行方向合并访存（coalesced），B 为列方向非合并访存（non-coalesced），B 侧带宽成为主要瓶颈。
 * 每线程对 A、B 各读取 K 次，总计 2*K 次全局访存，算术强度低、整体性能主要受全局内存带宽限制。
 *
 * 使用场景：
 * 小矩阵（M, N, K < 32）。
 * 正确性验证，以及与后续优化版本进行性能对比时作为基线实现。
 */
template <typename T>
__global__ void matmul_v0_naive_kernel(const T *__restrict__ a,
                                       const T *__restrict__ b,
                                       T *__restrict__ c,
                                       int M,
                                       int N,
                                       int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (likely(row < M && col < N))
    {
        T sum = 0;
        // 计算矩阵乘法的内积：C[row][col] = sum(A[row][k] * B[k][col])
        for (int k = 0; k < K; ++k)
        {
            // 访问模式：
            // a[row * K + k]: 行访问，内存合并（coalesced）
            // b[k * N + col]: 列访问，内存不合并（non-coalesced），性能瓶颈
            sum += a[row * K + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

// ============================================================================
// Version 1: 共享内存分块（Shared Memory Tiling）
// ============================================================================

/**
 * @brief Version 1: 使用共享内存分块的矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 这是第一个优化版本，使用共享内存缓存数据块：
 * 将K维度分成多个tile，每个tile大小为TILE_SIZE。
 * 每个线程块协作加载一个tile的数据到共享内存。
 * 利用共享内存的高带宽和低延迟特性。
 */
template <typename T>
__global__ void matmul_v1_tiled_kernel(const T *__restrict__ a,
                                       const T *__restrict__ b,
                                       T *__restrict__ c,
                                       int M,
                                       int N,
                                       int K)
{
    // 共享内存块大小：TILE_SIZE x TILE_SIZE
    // 这里默认使用 32x32 tile，兼顾线程数与共享内存占用
    constexpr int TILE_SIZE = 32;

    // 共享内存：每个线程块有TILE_SIZE x TILE_SIZE的共享内存
    // 用于缓存矩阵A和B的一个tile
    __shared__ T shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[TILE_SIZE][TILE_SIZE];

    // 计算当前线程负责的输出元素位置
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    // 分块计算：将K维度分成多个tile
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 阶段1：协作加载数据到共享内存
        // 每个线程负责加载一个元素
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        // 加载矩阵A的一个tile到共享内存
        // 注意：这里使用threadIdx.y和threadIdx.x来索引共享内存
        // 这样可以保证同一行的线程访问连续的内存地址
        if (likely(row < M && a_col < K))
            shared_a[threadIdx.y][threadIdx.x] = a[row * K + a_col];
        else
            shared_a[threadIdx.y][threadIdx.x] = 0;

        // 加载矩阵B的一个tile到共享内存
        if (likely(b_row < K && col < N))
            shared_b[threadIdx.y][threadIdx.x] = b[b_row * N + col];
        else
            shared_b[threadIdx.y][threadIdx.x] = 0;

        // 同步：确保所有线程都完成了数据加载
        __syncthreads();

        // 阶段2：从共享内存读取数据并计算部分和
        // 现在所有需要的数据都在共享内存中，可以高效访问
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            // 注意在当前的实现中有潜在的 bank 冲突问题，也是 V2 的优化需要解决的问题
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        // 同步：确保所有线程都完成了计算，才能加载下一个tile
        __syncthreads();
    }

    // 写入结果到全局内存
    if (likely(row < M && col < N))
    {
        c[row * N + col] = sum;
    }
}

// ============================================================================
// Version 2: 优化共享内存访问（Bank Conflict Avoidance）
// ============================================================================

/**
 * @brief Version 2: 避免共享内存bank冲突的矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 在Version 1的基础上，优化共享内存访问模式以避免bank冲突：
 * 对于大多数现代GPU,共享内存被组织成32个bank。一个 warp 包含 32 个线程，它们以 SIMT 方式同步执行。
 * Bank conflict 的检测和影响发生在同一 warp 内：当同一 warp 中的多个线程同时访问同一 bank
 * 的不同地址时，这些访问会被串行化，导致性能下降。 通过添加padding（+1列），改变内存布局，避免bank冲突。
 *
 * 在Version 1中，当多个线程访问shared_b[k][threadIdx.x]时，
 * 如果threadIdx.x相同但k不同，可能访问同一个bank。
 *
 * 优化方法：
 * 将共享内存数组从 [TILE_SIZE][TILE_SIZE] 改为 [TILE_SIZE][TILE_SIZE + 1]。
 * 这样同一行的相邻元素会映射到不同的bank。
 * 消除bank冲突，提高共享内存带宽利用率。
 *
 */
template <typename T>
__global__ void matmul_v2_bank_conflict_free_kernel(const T *__restrict__ a,
                                                    const T *__restrict__ b,
                                                    T *__restrict__ c,
                                                    int M,
                                                    int N,
                                                    int K)
{
    constexpr int TILE_SIZE = 32;

    // 共享内存：添加padding（+1列）以避免bank冲突
    // shared_a: [TILE_SIZE][TILE_SIZE] - 访问模式是行访问，通常不会有bank冲突
    // shared_b: [TILE_SIZE][TILE_SIZE + 1] - 添加padding避免bank冲突
    __shared__ T shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[TILE_SIZE][TILE_SIZE + 1];  // +1 padding避免bank冲突

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    T sum = 0;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 加载数据到共享内存（与Version 1相同）
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;

        if (likely(row < M && a_col < K))
            shared_a[threadIdx.y][threadIdx.x] = a[row * K + a_col];
        else
            shared_a[threadIdx.y][threadIdx.x] = 0;

        if (likely(b_row < K && col < N))
            // 注意：shared_b使用padding，但索引方式不变
            shared_b[threadIdx.y][threadIdx.x] = b[b_row * N + col];
        else
            shared_b[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // 计算部分和：现在访问shared_b时不会有bank冲突
        // 因为padding改变了内存布局，同一行的相邻元素映射到不同的bank
        for (int k = 0; k < TILE_SIZE; ++k)
        {
            sum += shared_a[threadIdx.y][k] * shared_b[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (likely(row < M && col < N))
    {
        c[row * N + col] = sum;
    }
}

// ============================================================================
// Version 3: 寄存器分块（Register Tiling）
// ============================================================================

/**
 * @brief Version 3: 使用寄存器分块的矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 在Version 2的基础上，使用寄存器分块让每个线程计算多个输出元素：
 * 每个线程计算 TILE_M x TILE_N 个输出元素（而不是1个）。
 * 利用寄存器存储中间结果，减少共享内存访问。
 * 提高计算强度（compute intensity），更好地利用计算资源。
 *
 * 优化原理：
 * 寄存器访问延迟：~1 cycle（最低延迟）。
 * 寄存器带宽：极高（几乎无限制）。
 * 通过让每个线程计算多个元素，增加计算/内存访问比。
 * 减少共享内存访问次数：从 O(TILE_SIZE) 降低到 O(TILE_SIZE / TILE_M)。
 *
 */
template <typename T>
__global__ void matmul_v3_register_tiling_kernel(const T *__restrict__ a,
                                                 const T *__restrict__ b,
                                                 T *__restrict__ c,
                                                 int M,
                                                 int N,
                                                 int K)
{
    const int TILE_SIZE = 32;
    const int TILE_M    = 4;  // 每个线程计算4行
    const int TILE_N    = 4;  // 每个线程计算4列

    __shared__ T shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[TILE_SIZE][TILE_SIZE + 1];

    // 计算当前线程负责的输出块的位置
    int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * TILE_N;

    // 寄存器数组：存储每个线程计算的多个输出元素
    T reg_c[TILE_M][TILE_N];
    // 初始化寄存器数组为0
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            reg_c[i][j] = 0;
        }
    }

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 加载数据到共享内存
        // 每个线程需要加载TILE_M行和TILE_N列的数据
        // 协作加载：所有线程一起加载TILE_SIZE x TILE_SIZE的数据块
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
                int load_row = threadIdx.y * TILE_M + i;
                int load_col = threadIdx.x * TILE_N + j;
                int a_col    = tile * TILE_SIZE + load_col;
                int b_row    = tile * TILE_SIZE + load_row;

                // 加载矩阵A的一个tile
                if (likely(base_row + i < M && a_col < K))
                    shared_a[load_row][load_col] = a[(base_row + i) * K + a_col];
                else
                    shared_a[load_row][load_col] = 0;

                // 加载矩阵B的一个tile
                if (likely(b_row < K && base_col + j < N))
                    shared_b[load_row][load_col] = b[b_row * N + (base_col + j)];
                else
                    shared_b[load_row][load_col] = 0;
            }
        }

        __syncthreads();

        // 计算部分和：每个线程计算 TILE_M x TILE_N 个元素
        // 使用寄存器存储中间结果，减少共享内存访问
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    // 从共享内存读取，累加到寄存器
                    int shared_row = threadIdx.y * TILE_M + i;
                    int shared_col = threadIdx.x * TILE_N + j;
                    reg_c[i][j] += shared_a[shared_row][k] * shared_b[k][shared_col];
                }
            }
        }

        __syncthreads();
    }

    // 将寄存器中的结果写回全局内存
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int row = base_row + i;
            int col = base_col + j;
            if (likely(row < M && col < N))
            {
                c[row * N + col] = reg_c[i][j];
            }
        }
    }
}

// ============================================================================
// Version 4: 向量化内存访问（Vectorized Memory Access）
// ============================================================================

/**
 * @brief Version 4: 使用向量化内存访问的矩阵乘法CUDA内核（仅支持float类型）
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 在Version 3的基础上，使用向量化内存访问提高内存带宽利用率：
 * 使用float4类型一次加载4个float值。
 * 减少内存事务数量，提高内存带宽利用率。
 * 要求数据对齐到16字节边界。
 *
 * 优化原理：
 * 现代GPU支持128位（16字节）的内存事务。
 * 使用float4可以一次加载4个float（16字节），充分利用128位事务。
 * 减少内存事务数量：从 N 次降低到 N/4 次。
 *
 * 注意事项：
 * 目前只支持float类型（float4）。
 * 要求数据对齐到16字节边界。
 * K必须是4的倍数（或者需要处理边界情况）。
 */
__global__ void matmul_v4_vectorized_kernel(const float *__restrict__ a,
                                            const float *__restrict__ b,
                                            float *__restrict__ c,
                                            int M,
                                            int N,
                                            int K)
{
    const int TILE_SIZE   = 32;
    const int TILE_M      = 4;
    const int TILE_N      = 4;
    const int VECTOR_SIZE = 4;  // float4 = 4个float

    // 只有当行跨度 / 列跨度本身是 4 的倍数时，才保证
    // (row * stride + col) * sizeof(float) 对齐到 16 字节，可以安全使用 float4
    const bool can_vectorize_a = (K % VECTOR_SIZE == 0);
    const bool can_vectorize_b = (N % VECTOR_SIZE == 0);

    __shared__ float shared_a[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_b[TILE_SIZE][TILE_SIZE + 1];

    int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * TILE_N;

    float reg_c[TILE_M][TILE_N];
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            reg_c[i][j] = 0;
        }
    }

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 向量化加载：使用float4一次加载4个float
        // 每个线程加载TILE_M行和TILE_N列的数据
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < TILE_N; j += VECTOR_SIZE)
            {
                int load_row = threadIdx.y * TILE_M + i;
                int load_col = threadIdx.x * TILE_N + j;
                int a_col    = tile * TILE_SIZE + load_col;
                int b_row    = tile * TILE_SIZE + load_row;

                // 加载矩阵A：使用float4向量化加载
                // 需要同时满足：
                // 1) can_vectorize_a: K 是 4 的倍数，保证行跨度对齐
                // 2) a_col 对齐到 4 的倍数
                // 3) 不越界（a_col + VECTOR_SIZE - 1 < K）
                if (likely(can_vectorize_a && base_row + i < M && a_col + VECTOR_SIZE - 1 < K &&
                           (a_col % VECTOR_SIZE == 0)))
                {
                    // 使用float4一次加载4个float
                    float4 vec_a = *reinterpret_cast<const float4 *>(&a[(base_row + i) * K + a_col]);
                    shared_a[load_row][load_col + 0] = vec_a.x;
                    shared_a[load_row][load_col + 1] = vec_a.y;
                    shared_a[load_row][load_col + 2] = vec_a.z;
                    shared_a[load_row][load_col + 3] = vec_a.w;
                }
                else
                {
                    // 边界情况：逐个加载
                    for (int v = 0; v < VECTOR_SIZE && j + v < TILE_N; ++v)
                    {
                        if (likely(base_row + i < M && a_col + v < K))
                            shared_a[load_row][load_col + v] = a[(base_row + i) * K + a_col + v];
                        else
                            shared_a[load_row][load_col + v] = 0;
                    }
                }

                // 加载矩阵B：使用float4向量化加载
                // 同理，只有当 N 是 4 的倍数且 base_col + j 按 4 对齐时才使用 float4
                if (likely(can_vectorize_b && b_row < K && base_col + j + VECTOR_SIZE - 1 < N &&
                           ((base_col + j) % VECTOR_SIZE == 0)))
                {
                    float4 vec_b                     = *reinterpret_cast<const float4 *>(&b[b_row * N + base_col + j]);
                    shared_b[load_row][load_col + 0] = vec_b.x;
                    shared_b[load_row][load_col + 1] = vec_b.y;
                    shared_b[load_row][load_col + 2] = vec_b.z;
                    shared_b[load_row][load_col + 3] = vec_b.w;
                }
                else
                {
                    // 边界情况：逐个加载
                    for (int v = 0; v < VECTOR_SIZE && j + v < TILE_N; ++v)
                    {
                        if (likely(b_row < K && base_col + j + v < N))
                            shared_b[load_row][load_col + v] = b[b_row * N + base_col + j + v];
                        else
                            shared_b[load_row][load_col + v] = 0;
                    }
                }
            }
        }

        __syncthreads();

        // 计算部分和
#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    int shared_row = threadIdx.y * TILE_M + i;
                    int shared_col = threadIdx.x * TILE_N + j;
                    reg_c[i][j] += shared_a[shared_row][k] * shared_b[k][shared_col];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果（与Version 3相同）
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int row = base_row + i;
            int col = base_col + j;
            if (likely(row < M && col < N))
            {
                c[row * N + col] = reg_c[i][j];
            }
        }
    }
}

// ============================================================================
// Version 5: 双缓冲（Double Buffering）
// ============================================================================

/**
 * @brief Version 5: 使用双缓冲的矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 在Version 3的基础上，使用双缓冲技术重叠计算和内存加载：
 * 使用两个共享内存缓冲区（buffer 0和buffer 1）。
 * 当一个缓冲区用于计算时，另一个缓冲区用于加载下一个tile的数据。
 *
 * 优化1（GMEM 预取）：下一 tile 先从 GMEM 读到寄存器 ldg_a/ldg_b，再做当前 tile 计算，
 * 算完后将寄存器写回 SMEM，使 GMEM 加载延迟与计算重叠。
 *
 * 优化2（内层 k 寄存器双缓冲）：内层 k 循环中用 a_frag[2][TILE_M]、b_frag[2][TILE_N]
 * 双缓冲，每次迭代预取下一 k 的 SMEM 到寄存器再计算当前 k，使 SMEM->reg 传输 与乘加重叠。
 *
 * 实现方法：
 * 使用两个共享内存数组 shared_a[2][TILE_SIZE][TILE_SIZE]、shared_b[2][TILE_SIZE][TILE_SIZE+1]。
 * 每轮顺序：预取下一 tile 到寄存器 -> 用当前 buffer 计算（含 k 维寄存器双缓冲）-> 写回 SMEM -> sync。
 *
 */
template <typename T>
__global__ void matmul_v5_double_buffering_kernel(const T *__restrict__ a,
                                                  const T *__restrict__ b,
                                                  T *__restrict__ c,
                                                  int M,
                                                  int N,
                                                  int K)
{
    const int TILE_SIZE = 32;
    const int TILE_M    = 4;
    const int TILE_N    = 4;

    // 双缓冲：使用两个共享内存缓冲区
    __shared__ T shared_a[2][TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[2][TILE_SIZE][TILE_SIZE + 1];

    int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * TILE_N;

    T reg_c[TILE_M][TILE_N];
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            reg_c[i][j] = 0;
        }
    }

    // 双缓冲用到的寄存器（两层）：
    // 1) ldg_a/ldg_b：下一 tile 从 GMEM 预取到这里，本轮算完再写回 SMEM
    // 2) a_frag/b_frag：内层 k 循环里，当前/下一 k 的 A 行、B 列，SMEM->reg 传输 与乘加重叠
    T ldg_a[TILE_M][TILE_N]; // (load global memory to register)
    T ldg_b[TILE_M][TILE_N];
    T a_frag[2][TILE_M];
    T b_frag[2][TILE_N];

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 流水启动前：先把第 0 个 tile 放进 shared，供第一轮 Step 2 使用
    int tile = 0;
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int load_row = threadIdx.y * TILE_M + i;
            int load_col = threadIdx.x * TILE_N + j;
            int a_col    = tile * TILE_SIZE + load_col;
            int b_row    = tile * TILE_SIZE + load_row;

            if (likely(base_row + i < M && a_col < K))
                shared_a[0][load_row][load_col] = a[(base_row + i) * K + a_col];
            else
                shared_a[0][load_row][load_col] = 0;

            if (likely(b_row < K && base_col + j < N))
                shared_b[0][load_row][load_col] = b[b_row * N + (base_col + j)];
            else
                shared_b[0][load_row][load_col] = 0;
        }
    }

    __syncthreads();

    // 主循环内双缓冲每轮迭代的三个步骤：
    //   Step 1: 预取下一 tile，GMEM -> 寄存器 ldg_a/ldg_b（不写 SMEM）
    //   Step 2: 用当前 tile 计算，即 shared_a[curr_buffer]/shared_b[curr_buffer] -> reg_c
    //           内层 k 循环再用 a_frag/b_frag 做 SMEM->reg 传输 与乘加的重叠
    //   Step 3: 将 Step 1 预取的数据从寄存器写回 SMEM，下一轮该 buffer 作为 curr_buffer 供 Step 2 使用
    // 重叠的含义是：Step 1 预取的 GMEM 加载延迟，主要被本轮及后续轮次的 Step 2 计算时间隐藏（通过 warp 调度和流水覆盖）
    for (tile = 0; tile < num_tiles; ++tile)
    {
        int curr_buffer = tile % 2;  // 当前 tile 在 shared 的槽位
        int next_buffer = (tile + 1) % 2;  // 下一 tile 要写回的槽位

        // Step 1：预取下一 tile，GMEM -> 寄存器 ldg_a / ldg_b（不写 SMEM）
        // 本迭代稍后的 Step 2 会与这批 load 的延迟在时间上重叠
        if (likely(tile + 1 < num_tiles))
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    int load_row   = threadIdx.y * TILE_M + i;
                    int load_col   = threadIdx.x * TILE_N + j;
                    int next_a_col = (tile + 1) * TILE_SIZE + load_col;
                    int next_b_row = (tile + 1) * TILE_SIZE + load_row;

                    if (likely(base_row + i < M && next_a_col < K))
                        ldg_a[i][j] = a[(base_row + i) * K + next_a_col];
                    else
                        ldg_a[i][j] = 0;

                    if (likely(next_b_row < K && base_col + j < N))
                        ldg_b[i][j] = b[next_b_row * N + (base_col + j)];
                    else
                        ldg_b[i][j] = 0;
                }
            }
        }

        // Step 2：用当前 tile（已在 shared_a[curr_buffer] / shared_b[curr_buffer]）计算
        // 内层 k 再做一层寄存器双缓冲：先预取下一 k 到 frag，再算当前 k，以隐藏 SMEM 读延迟
        // 预取 k=0 的 A 行、B 列到 a_frag[0] / b_frag[0]
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
            int shared_row = threadIdx.y * TILE_M + i;
            a_frag[0][i] = shared_a[curr_buffer][shared_row][0];
        }
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int shared_col = threadIdx.x * TILE_N + j;
            b_frag[0][j] = shared_b[curr_buffer][0][shared_col];
        }

#pragma unroll
        for (int k = 0; k < TILE_SIZE - 1; ++k)
        {
            // 预取 k+1 的 A 行、B 列到 a_frag[(k+1)%2] / b_frag[(k+1)%2]
            // 对整个 SM 上所有 warp：当某个 warp 在做 “预取 k+1” 的 load 时，
            // 如果被 SMEM 延迟挡住，调度器会把别的 warp 拿出来跑，就不会出现“所有 warp 都一条 load 卡死在那”的极端情况。
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
                int shared_row = threadIdx.y * TILE_M + i;
                a_frag[(k + 1) % 2][i] = shared_a[curr_buffer][shared_row][k + 1];
            }
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
                int shared_col = threadIdx.x * TILE_N + j;
                b_frag[(k + 1) % 2][j] = shared_b[curr_buffer][k + 1][shared_col];
            }
            // 用当前 k 的 a_frag[k%2] / b_frag[k%2] 做乘加
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    reg_c[i][j] += a_frag[k % 2][i] * b_frag[k % 2][j];
                }
            }
        }
        // k = TILE_SIZE-1 的乘加（frag 已在上一轮预取）
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
                reg_c[i][j] += a_frag[(TILE_SIZE - 1) % 2][i] * b_frag[(TILE_SIZE - 1) % 2][j];
            }
        }

        // Step 3：将本迭代 Step 1 预取的下一 tile 从寄存器写回 SMEM，下一轮将作为 curr_buffer 供 Step 2 使用
        if (likely(tile + 1 < num_tiles))
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    int load_row = threadIdx.y * TILE_M + i;
                    int load_col = threadIdx.x * TILE_N + j;
                    shared_a[next_buffer][load_row][load_col] = ldg_a[i][j];
                    shared_b[next_buffer][load_row][load_col] = ldg_b[i][j];
                }
            }
            __syncthreads();
        }
    }

    // 写回结果
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int row = base_row + i;
            int col = base_col + j;
            if (likely(row < M && col < N))
            {
                c[row * N + col] = reg_c[i][j];
            }
        }
    }
}

// ============================================================================
// Version 6: 循环展开（Loop Unrolling）
// ============================================================================

/**
 * @brief Version 6: 在 Version 5 基础上对内层 k 循环做手动展开的矩阵乘法 CUDA 内核
 * @tparam T 数据类型
 * @tparam TILE_SIZE tile 边长（默认 32）
 * @tparam TILE_M 每个线程在 M 方向计算的元素数（默认 4）
 * @tparam TILE_N 每个线程在 N 方向计算的元素数（默认 4）
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 在 Version 5 的基础上，保持相同的双缓冲与数据流（Step 1/2/3、ldg_a/ldg_b、a_frag/b_frag），
 * 仅对内层 k 循环按 UNROLL_FACTOR=4 做手动展开：每次外层迭代处理 4 个连续 k，
 * 减少循环控制与分支，提高指令级并行度（ILP），便于编译器调度。
 *
 * 该内核通过模板参数 TILE_SIZE / TILE_M / TILE_N 支持不同的 tile 配置，
 * 默认配置为 32x32 tile，单线程 4x4 子块。
 */
template <typename T, int TILE_SIZE = 32, int TILE_M = 4, int TILE_N = 4>
__global__ void matmul_v6_unrolled_kernel(const T *__restrict__ a,
                                          const T *__restrict__ b,
                                          T *__restrict__ c,
                                          int M,
                                          int N,
                                          int K)
{
    static_assert(TILE_SIZE % TILE_M == 0, "TILE_SIZE must be divisible by TILE_M");
    static_assert(TILE_SIZE % TILE_N == 0, "TILE_SIZE must be divisible by TILE_N");
    constexpr int UNROLL_FACTOR = 4;

    __shared__ T shared_a[2][TILE_SIZE][TILE_SIZE];
    __shared__ T shared_b[2][TILE_SIZE][TILE_SIZE + 1];

    int base_row = blockIdx.y * TILE_SIZE + threadIdx.y * TILE_M;
    int base_col = blockIdx.x * TILE_SIZE + threadIdx.x * TILE_N;

    T reg_c[TILE_M][TILE_N];
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            reg_c[i][j] = 0;
        }
    }

    T ldg_a[TILE_M][TILE_N];
    T ldg_b[TILE_M][TILE_N];
    T a_frag[2][TILE_M];
    T b_frag[2][TILE_N];

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // 流水启动前：第 0 个 tile 放进 shared，供第一轮 Step 2 使用
    int tile = 0;
#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int load_row = threadIdx.y * TILE_M + i;
            int load_col = threadIdx.x * TILE_N + j;
            int a_col    = tile * TILE_SIZE + load_col;
            int b_row    = tile * TILE_SIZE + load_row;

            if (likely(base_row + i < M && a_col < K))
                shared_a[0][load_row][load_col] = a[(base_row + i) * K + a_col];
            else
                shared_a[0][load_row][load_col] = 0;

            if (likely(b_row < K && base_col + j < N))
                shared_b[0][load_row][load_col] = b[b_row * N + (base_col + j)];
            else
                shared_b[0][load_row][load_col] = 0;
        }
    }

    __syncthreads();

    for (tile = 0; tile < num_tiles; ++tile)
    {
        int curr_buffer = tile % 2;
        int next_buffer = (tile + 1) % 2;

        // Step 1：预取下一 tile 到 ldg_a / ldg_b
        if (likely(tile + 1 < num_tiles))
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    int load_row   = threadIdx.y * TILE_M + i;
                    int load_col   = threadIdx.x * TILE_N + j;
                    int next_a_col = (tile + 1) * TILE_SIZE + load_col;
                    int next_b_row = (tile + 1) * TILE_SIZE + load_row;

                    if (likely(base_row + i < M && next_a_col < K))
                        ldg_a[i][j] = a[(base_row + i) * K + next_a_col];
                    else
                        ldg_a[i][j] = 0;

                    if (likely(next_b_row < K && base_col + j < N))
                        ldg_b[i][j] = b[next_b_row * N + (base_col + j)];
                    else
                        ldg_b[i][j] = 0;
                }
            }
        }

        // Step 2：用当前 tile 计算，内层 k 按 UNROLL_FACTOR=4 手动展开，仍用 a_frag/b_frag 双缓冲
#pragma unroll
        for (int i = 0; i < TILE_M; ++i)
        {
            int shared_row = threadIdx.y * TILE_M + i;
            a_frag[0][i] = shared_a[curr_buffer][shared_row][0];
        }
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int shared_col = threadIdx.x * TILE_N + j;
            b_frag[0][j] = shared_b[curr_buffer][0][shared_col];
        }

        for (int k_base = 0; k_base < TILE_SIZE - 1; k_base += UNROLL_FACTOR)
        {
            // 手动展开 4 步：每步预取下一 k 到 frag，再用当前 k 的 frag 乘加
            int k0 = k_base;
            int k1 = k_base + 1;
            int k2 = k_base + 2;
            int k3 = k_base + 3;

#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
                int shared_row = threadIdx.y * TILE_M + i;
                a_frag[(k0 + 1) % 2][i] = shared_a[curr_buffer][shared_row][k1];
            }
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
                int shared_col = threadIdx.x * TILE_N + j;
                b_frag[(k0 + 1) % 2][j] = shared_b[curr_buffer][k1][shared_col];
            }
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    reg_c[i][j] += a_frag[k0 % 2][i] * b_frag[k0 % 2][j];
                }
            }

#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
                int shared_row = threadIdx.y * TILE_M + i;
                a_frag[(k1 + 1) % 2][i] = shared_a[curr_buffer][shared_row][k2];
            }
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
                int shared_col = threadIdx.x * TILE_N + j;
                b_frag[(k1 + 1) % 2][j] = shared_b[curr_buffer][k2][shared_col];
            }
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    reg_c[i][j] += a_frag[k1 % 2][i] * b_frag[k1 % 2][j];
                }
            }

#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
                int shared_row = threadIdx.y * TILE_M + i;
                a_frag[(k2 + 1) % 2][i] = shared_a[curr_buffer][shared_row][k3];
            }
#pragma unroll
            for (int j = 0; j < TILE_N; ++j)
            {
                int shared_col = threadIdx.x * TILE_N + j;
                b_frag[(k2 + 1) % 2][j] = shared_b[curr_buffer][k3][shared_col];
            }
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    reg_c[i][j] += a_frag[k2 % 2][i] * b_frag[k2 % 2][j];
                }
            }

            // 第 4 步：仅当 k3+1 < TILE_SIZE 时预取 k3+1；然后算 k3
            if (k3 + 1 < TILE_SIZE)
            {
#pragma unroll
                for (int i = 0; i < TILE_M; ++i)
                {
                    int shared_row = threadIdx.y * TILE_M + i;
                    a_frag[(k3 + 1) % 2][i] = shared_a[curr_buffer][shared_row][k3 + 1];
                }
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    int shared_col = threadIdx.x * TILE_N + j;
                    b_frag[(k3 + 1) % 2][j] = shared_b[curr_buffer][k3 + 1][shared_col];
                }
            }
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    reg_c[i][j] += a_frag[k3 % 2][i] * b_frag[k3 % 2][j];
                }
            }
        }
        // 内层 k 已全部在展开循环中完成（含 k=TILE_SIZE-1），无需再补一次乘加

        // Step 3：写回 ldg 到 SMEM
        if (likely(tile + 1 < num_tiles))
        {
#pragma unroll
            for (int i = 0; i < TILE_M; ++i)
            {
#pragma unroll
                for (int j = 0; j < TILE_N; ++j)
                {
                    int load_row = threadIdx.y * TILE_M + i;
                    int load_col = threadIdx.x * TILE_N + j;
                    shared_a[next_buffer][load_row][load_col] = ldg_a[i][j];
                    shared_b[next_buffer][load_row][load_col] = ldg_b[i][j];
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int i = 0; i < TILE_M; ++i)
    {
#pragma unroll
        for (int j = 0; j < TILE_N; ++j)
        {
            int row = base_row + i;
            int col = base_col + j;
            if (likely(row < M && col < N))
            {
                c[row * N + col] = reg_c[i][j];
            }
        }
    }
}

// ============================================================================
// Version 7: Warptiling（Warp-level Tiling）
// ============================================================================

/**
 * @brief Version 7: 使用Warptiling的矩阵乘法CUDA内核
 * @tparam T 数据类型
 * @tparam BM Block级别的M维度tile大小
 * @tparam BN Block级别的N维度tile大小
 * @tparam BK Block级别的K维度tile大小
 * @tparam WM Warp级别的M维度tile大小
 * @tparam WN Warp级别的N维度tile大小
 * @tparam TM Thread级别的M维度tile大小
 * @tparam TN Thread级别的N维度tile大小
 * @param a 输入矩阵A (M x K)
 * @param b 输入矩阵B (K x N)
 * @param c 输出矩阵C (M x N)
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * 在Version 6的基础上，引入warp-level tiling：
 * 三层tiling：Block -> Warp -> Thread。
 * 每个warp负责一个子块，更好地利用warp局部性。
 * 减少shared memory bank冲突。
 * 提高寄存器重用率。
 *
 * 优化原理：
 * Warp是GPU执行的基本单位（32个线程）。
 * Warp内线程可以高效协作，共享数据。
 * Warp-level tiling可以更好地利用warp内的数据局部性。
 * 减少跨warp的shared memory访问冲突。
 *
 * 适用场景：
 * 超大矩阵
 */
template <typename T, int BM = 128, int BN = 128, int BK = 16, int WM = 64, int WN = 64, int TM = 8, int TN = 16>
__global__ void matmul_v7_warptiling_kernel(const T *__restrict__ a,
                                            const T *__restrict__ b,
                                            T *__restrict__ c,
                                            int M,
                                            int N,
                                            int K)
{
    // 静态断言：确保参数合理性
    static_assert(BM % WM == 0, "BM must be divisible by WM");
    static_assert(BN % WN == 0, "BN must be divisible by WN");
    static_assert(WM % TM == 0, "WM must be divisible by TM");
    static_assert(WN % TN == 0, "WN must be divisible by TN");
    static_assert((BM * BN) % (WM * WN) == 0, "Block size must be divisible by warp tile size");

    // 共享内存：block级别的tile，添加padding避免bank冲突
    __shared__ T shared_a[BM][BK + 1];
    __shared__ T shared_b[BK][BN + 1];

    // Warp和thread索引
    const int warp_id = threadIdx.y;
    const int lane_id = threadIdx.x;

    // 计算当前warp在block中的位置
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    // 计算当前thread在warp中的位置
    const int thread_row = lane_id / (WN / TN);
    const int thread_col = lane_id % (WN / TN);

    // 计算当前thread负责的输出块在全局矩阵中的位置
    const int base_row = blockIdx.y * BM + warp_row * WM + thread_row * TM;
    const int base_col = blockIdx.x * BN + warp_col * WN + thread_col * TN;

    // 寄存器数组：存储每个thread计算的多个输出元素
    T reg_c[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            reg_c[i][j] = 0;
        }
    }

    // 寄存器：存储从shared memory加载的数据
    T reg_a[TM];
    T reg_b[TN];

    int num_tiles = (K + BK - 1) / BK;
    for (int tile = 0; tile < num_tiles; ++tile)
    {
        // 协作加载：所有thread协作加载block级别的tile到shared memory
        // 每个thread加载多个元素以提高效率
        const int num_threads         = blockDim.x * blockDim.y;
        const int elements_per_thread = (BM * BK + num_threads - 1) / num_threads;

        for (int i = 0; i < elements_per_thread; ++i)
        {
            int idx = (threadIdx.y * blockDim.x + threadIdx.x) * elements_per_thread + i;
            if (idx < BM * BK)
            {
                int row        = idx / BK;
                int col        = idx % BK;
                int global_row = blockIdx.y * BM + row;
                int global_col = tile * BK + col;
                if (likely(global_row < M && global_col < K))
                {
                    shared_a[row][col] = a[global_row * K + global_col];
                }
                else
                {
                    shared_a[row][col] = 0;
                }
            }
        }

        // 加载矩阵B
        const int b_elements_per_thread = (BK * BN + num_threads - 1) / num_threads;
        for (int i = 0; i < b_elements_per_thread; ++i)
        {
            int idx = (threadIdx.y * blockDim.x + threadIdx.x) * b_elements_per_thread + i;
            if (idx < BK * BN)
            {
                int row        = idx / BN;
                int col        = idx % BN;
                int global_row = tile * BK + row;
                int global_col = blockIdx.x * BN + col;
                if (likely(global_row < K && global_col < N))
                {
                    shared_b[row][col] = b[global_row * N + global_col];
                }
                else
                {
                    shared_b[row][col] = 0;
                }
            }
        }

        __syncthreads();

        // 计算部分和：三层循环对应三层tiling
        for (int k = 0; k < BK; ++k)
        {
            // 从shared memory加载到寄存器（warp级别）
            // 每个thread加载自己需要的TM行和TN列的数据
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                int shared_row = warp_row * WM + thread_row * TM + i;
                reg_a[i]       = shared_a[shared_row][k];
            }

#pragma unroll
            for (int j = 0; j < TN; ++j)
            {
                int shared_col = warp_col * WN + thread_col * TN + j;
                reg_b[j]       = shared_b[k][shared_col];
            }

            // 计算：使用寄存器中的数据，提高ILP
#pragma unroll
            for (int i = 0; i < TM; ++i)
            {
#pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // 写回结果到全局内存
#pragma unroll
    for (int i = 0; i < TM; ++i)
    {
#pragma unroll
        for (int j = 0; j < TN; ++j)
        {
            int row = base_row + i;
            int col = base_col + j;
            if (likely(row < M && col < N))
            {
                c[row * N + col] = reg_c[i][j];
            }
        }
    }
}





/**
 * @brief 启动Version 0: 朴素矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v0_naive_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_v0_naive_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 1: 共享内存分块矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v1_tiled_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v1_tiled_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 2: 避免bank冲突的矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v2_bank_conflict_free_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    constexpr int TILE_SIZE = 32;
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v2_bank_conflict_free_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 3: 寄存器分块矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v3_register_tiling_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    /*
    jinbo@JinboBook:~/gitme/C_OriginDL$ ./build/bin/benchmark/bench_mat_mul -d cuda -w 2 -r 10 -s
    10000,10000:10000,10000 shape                                   repeat          device          dtype
    origindl_time_us {10000, 10000}:{10000, 10000}           10              cuda:0          float32         4.3000
    jinbo@JinboBook:~/gitme/C_OriginDL$ ./build/bin/benchmark/bench_mat_mul -d cuda -w 2 -r 10 -s
    10000,10000:10000,10000 shape                                   repeat          device          dtype
    origindl_time_us {10000, 10000}:{10000, 10000}           10              cuda:0          float32         633912.8000
    */
    const int TILE_SIZE =
        32;  // 从 16 改为 32，性能提升 2.63 倍, 在 16 的情况下，每个 block 只有 16 个线程，没有打满一个 warp。
    const int TILE_M = 4;
    const int TILE_N = 4;
    // 注意：每个线程计算TILE_M x TILE_N个元素，所以block大小需要相应调整
    dim3 block(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v3_register_tiling_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 4: 向量化矩阵乘法内核（仅float类型）
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
void launch_matmul_v4_vectorized_kernel(const float *a, const float *b, float *c, int M, int N, int K)
{
    const int TILE_SIZE = 32;
    const int TILE_M    = 4;
    const int TILE_N    = 4;
    dim3 block(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v4_vectorized_kernel<<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 5: 双缓冲矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v5_double_buffering_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    const int TILE_SIZE = 32;
    const int TILE_M    = 4;
    const int TILE_N    = 4;
    dim3 block(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v5_double_buffering_kernel<T><<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 6: 循环展开矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T, int TILE_SIZE, int TILE_M, int TILE_N>
void launch_matmul_v6_unrolled_kernel_config(const T *a, const T *b, T *c, int M, int N, int K)
{
    dim3 block(TILE_SIZE / TILE_N, TILE_SIZE / TILE_M);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    matmul_v6_unrolled_kernel<T, TILE_SIZE, TILE_M, TILE_N><<<grid, block>>>(a, b, c, M, N, K);
}

template <typename T>
void launch_matmul_v6_unrolled_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    // 默认配置：32x32 tile，单线程 4x4 子块
    constexpr int TILE_SIZE = 32;
    constexpr int TILE_M    = 4;
    constexpr int TILE_N    = 4;
    launch_matmul_v6_unrolled_kernel_config<T, TILE_SIZE, TILE_M, TILE_N>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 7: Warptiling矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v7_warptiling_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    // 使用默认参数：BM=128, BN=128, BK=16, WM=64, WN=64, TM=8, TN=8
    // Block大小：每个block有 (BN/WN) * (BM/WM) = 2 * 2 = 4个warp
    // 每个warp有32个thread，所以block有128个thread
    // block配置：blockDim.x = 32 (每个warp的线程数), blockDim.y = 4 (warp数量)
    constexpr int BM                  = 128;
    constexpr int BN                  = 128;
    constexpr int WM                  = 64;
    constexpr int WN                  = 64;
    constexpr int num_warps_per_block = (BM / WM) * (BN / WN);  // 4个warp
    constexpr int threads_per_warp    = 32;

    dim3 block(threads_per_warp, num_warps_per_block);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    matmul_v7_warptiling_kernel<T, BM, BN, 16, WM, WN, 8, 16><<<grid, block>>>(a, b, c, M, N, K);
}

/**
 * @brief 启动Version 9: 自动调优矩阵乘法内核
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 */
template <typename T>
void launch_matmul_v9_autotuning_kernel(const T *a, const T *b, T *c, int M, int N, int K)
{
    // 根据矩阵大小自动选择最优配置
    // 这里使用简化的选择逻辑，实际应用中可以通过autotuning找到最优配置
    if (M >= 2048 && N >= 2048 && K >= 2048)
    {
        // 超大矩阵：使用 V6 模板版本，配置较大的 tile
        // 注意：更大的 TILE_SIZE 需要考虑 shared memory 限制，这里先使用 32x32 配置，
        // 后续可以通过实际 benchmark 再调整 TILE_M / TILE_N。
        launch_matmul_v6_unrolled_kernel_config<T, 32, 4, 4>(a, b, c, M, N, K);
    }
    else if (M >= 1024 && N >= 1024 && K >= 1024)
    {
        // 大矩阵：同样使用 V6 模板版本（后续可根据 benchmark 调整子块尺寸）
        launch_matmul_v6_unrolled_kernel_config<T, 32, 4, 4>(a, b, c, M, N, K);
    }
    else if (M >= 512 && N >= 512 && K >= 512)
    {
        // 中等大矩阵：继续使用默认 V6 配置
        launch_matmul_v6_unrolled_kernel<T>(a, b, c, M, N, K);
    }
    else
    {
        // 小到中等矩阵：使用Version 6（循环展开）
        launch_matmul_v6_unrolled_kernel<T>(a, b, c, M, N, K);
    }
}

/**
 * @brief Version 6 自动配置辅助函数：根据矩阵尺寸为 V6 选择不同配置
 *
 * @details
 * 目前使用与 V9 启动函数相同的简单分段逻辑：
 * - M/N/K >= 2048: 为超大矩阵预留更 aggressive 的 tile 配置；
 * - M/N/K >= 1024: 大矩阵配置；
 * - M/N/K >= 512 : 中等大矩阵；
 * - 其他更小尺寸：使用默认 V6 配置。
 *
 * 后续可以根据 scripts/bench_matmul_algos.sh 的 benchmark 结果，针对不同区间调整
 * (TILE_SIZE, TILE_M, TILE_N) 参数。
 */
template <typename T>
inline void launch_matmul_v6_auto_config(const T *a, const T *b, T *c, int M, int N, int K)
{
    if (M >= 2048 && N >= 2048 && K >= 2048)
    {
        launch_matmul_v6_unrolled_kernel_config<T, 32, 4, 4>(a, b, c, M, N, K);
    }
    else if (M >= 1024 && N >= 1024 && K >= 1024)
    {
        launch_matmul_v6_unrolled_kernel_config<T, 32, 4, 4>(a, b, c, M, N, K);
    }
    else if (M >= 512 && N >= 512 && K >= 512)
    {
        launch_matmul_v6_unrolled_kernel<T>(a, b, c, M, N, K);
    }
    else
    {
        launch_matmul_v6_unrolled_kernel<T>(a, b, c, M, N, K);
    }
}

/**
 * @brief Version 4启动函数的辅助函数：处理float类型的特化
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 *
 * @details
 * Version 4只支持float类型，对于其他类型降级到Version 3
 */
template <typename T>
void launch_matmul_v4_or_fallback(const T *a, const T *b, T *c, int M, int N, int K)
{
    // 对于非float类型，降级到Version 3
    launch_matmul_v3_register_tiling_kernel<T>(a, b, c, M, N, K);
}

// float类型的特化：使用Version 4向量化版本
template <>
void launch_matmul_v4_or_fallback<float>(const float *a, const float *b, float *c, int M, int N, int K)
{
    launch_matmul_v4_vectorized_kernel(a, b, c, M, N, K);
}

/**
 * @brief 将数字转换为 MatMulVersion 枚举
 * @param algo_version 算法版本号（数字）
 * @return MatMulVersion 枚举值，如果无效则返回 V_AUTO
 *
 * @details
 * 支持的版本号：
 * 0: V0_NAIVE。
 * 1: V1_TILED。
 * 2: V2_BANK_CONFLICT_FREE。
 * 3: V3_REGISTER_TILING。
 * 4: V4_VECTORIZED。
 * 5: V5_DOUBLE_BUFFERING。
 * 6: V6_UNROLLED。
 * 7: V7_WARPTILING。
 * 6666: V_AUTO。
 * 其他值: 返回 V_AUTO（自动选择）。
 */
inline MatMulVersion algo_version_to_matmul_version(int32_t algo_version)
{
    switch (algo_version)
    {
        case 0:
            return MatMulVersion::V0_NAIVE;
        case 1:
            return MatMulVersion::V1_TILED;
        case 2:
            return MatMulVersion::V2_BANK_CONFLICT_FREE;
        case 3:
            return MatMulVersion::V3_REGISTER_TILING;
        case 4:
            return MatMulVersion::V4_VECTORIZED;
        case 5:
            return MatMulVersion::V5_DOUBLE_BUFFERING;
        case 6:
            return MatMulVersion::V6_UNROLLED;
        case 7:
            return MatMulVersion::V7_WARPTILING;
        case 6666:
            return MatMulVersion::V_AUTO;
        default:
            // 无效的算法版本号，自动调整到 V_AUTO
            return MatMulVersion::V_AUTO;
    }
}

/**
 * @brief 启动2D矩阵乘法内核（自动选择最优版本）
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param c 输出矩阵C
 * @param M A的行数
 * @param N B的列数
 * @param K A的列数和B的行数
 * @param version 矩阵乘法版本（默认使用最优版本）
 *
 * @details
 * 自动模式（V_AUTO 或无效版本）基于 benchmark 数据做简单的尺寸分段：
 * - 小矩阵（例如 max_dim <= 128）优先使用 Version 2（共享内存避免 bank 冲突）；
 * - 中等矩阵：统一走 Version 6，并通过 launch_matmul_v6_auto_config 根据尺寸选择配置；
 * - 大矩阵（例如 max_dim >= 4096 且形状不过于狭长）使用 Version 7（warp tiling）。
 *
 * 尺寸阈值与具体参数可根据 scripts/bench_matmul_algos.sh 的 benchmark 结果进一步调整。
 */
template <typename T>
void launch_matmul_2d_kernel(const T *a,
                             const T *b,
                             T *c,
                             int M,
                             int N,
                             int K,
                             MatMulVersion version = MatMulVersion::V_AUTO)
{
    switch (version)
    {
        case MatMulVersion::V0_NAIVE:
            launch_matmul_v0_naive_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V1_TILED:
            launch_matmul_v1_tiled_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V2_BANK_CONFLICT_FREE:
            launch_matmul_v2_bank_conflict_free_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V3_REGISTER_TILING:
            launch_matmul_v3_register_tiling_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V4_VECTORIZED:
            launch_matmul_v4_or_fallback<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V5_DOUBLE_BUFFERING:
            launch_matmul_v5_double_buffering_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V6_UNROLLED:
            launch_matmul_v6_unrolled_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V7_WARPTILING:
            launch_matmul_v7_warptiling_kernel<T>(a, b, c, M, N, K);
            break;
        case MatMulVersion::V_AUTO:
        default:
        {
            // 自动模式：根据矩阵尺寸在 V2 / V6 / V7 之间做简单分层
            const int max_dim = (M > N) ? ((M > K) ? M : K) : ((N > K) ? N : K);
            const int min_dim = (M < N) ? ((M < K) ? M : K) : ((N < K) ? N : K);

            // 小矩阵：Version 2 对小尺寸下的启动开销和共享内存访问更友好
            if (max_dim <= 128)
            {
                launch_matmul_v2_bank_conflict_free_kernel<T>(a, b, c, M, N, K);
                break;
            }

            // 形状比例，用于避免在极端长条矩阵上盲目选择 V7
            const float aspect = static_cast<float>(max_dim) / static_cast<float>(min_dim > 0 ? min_dim : 1);

            // 大矩阵且形状不过于狭长：使用 Version 7（warp tiling）
            if (max_dim >= 4096 && aspect <= 4.0f)
            {
                launch_matmul_v7_warptiling_kernel<T>(a, b, c, M, N, K);
            }
            else
            {
                // 其余中等尺寸统一走 Version 6，由辅助函数内部根据尺寸再细分配置
                launch_matmul_v6_auto_config<T>(a, b, c, M, N, K);
            }
            break;
        }
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

    // 从环境变量获取算法版本号
    int32_t algo_version  = utils::EnvConfig::GetInstance().kernel_algo();
    MatMulVersion version = algo_version_to_matmul_version(algo_version);

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
            launch_matmul_2d_kernel(a_ptr->data_ptr<T>(), b_ptr->data_ptr<T>(), result->data_ptr<T>(), M, N, K,
                                    version);
        }
#else
        launch_matmul_2d_kernel(a_ptr->data_ptr<T>(), b_ptr->data_ptr<T>(), result->data_ptr<T>(), M, N, K, version);
#endif
    });

    return result;
}

}  // namespace cuda
}  // namespace origin
