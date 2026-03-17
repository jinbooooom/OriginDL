#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

#include <type_traits>

namespace origin
{
namespace cuda
{

/**
 * @brief warp 内求和归约（使用洗牌指令）
 */
template <typename T>
__inline__ __device__ T warp_shuffle_reduce_sum(T val)
{
    unsigned mask = 0xffffffff;

    // 蝶形（xor）规约：经过 log2(warpSize) 轮后，warp 内所有线程的 val 都相同
    //
    // 以 warpSize = 8 为例（lane 0~7，每个线程初始有 a0~a7）：
    //
    // 第 1 轮 offset = 4（按 bit 2 做异或配对）：
    //   0^4=4, 1^4=5, 2^4=6, 3^4=7
    //   lane 0: a0 + a4    lane 1: a1 + a5    lane 2: a2 + a6    lane 3: a3 + a7
    //   lane 4: a4 + a0    lane 5: a5 + a1    lane 6: a6 + a2    lane 7: a7 + a3
    //
    // 第 2 轮 offset = 2（按 bit 1 做异或配对）：
    //   0^2=2, 1^2=3, 4^2=6, 5^2=7
    //   lane 0: (a0+a4) + (a2+a6)
    //   lane 1: (a1+a5) + (a3+a7)
    //   lane 2: (a2+a6) + (a0+a4)
    //   lane 3: (a3+a7) + (a1+a5)
    //   lane 4: (a4+a0) + (a6+a2)
    //   lane 5: (a5+a1) + (a7+a3)
    //   lane 6: (a6+a2) + (a4+a0)
    //   lane 7: (a7+a3) + (a5+a1)
    //
    // 第 3 轮 offset = 1（按 bit 0 做异或配对）：
    //   0^1=1, 2^1=3, 4^1=5, 6^1=7
    //   每个 lane 再与对应配对 lane 的“部分和”相加，最终 lane 0~7 全部都变成 (a0+a1+...+a7)
    //
    // 对 warpSize = 32 也是同理，只是 offset 顺序是 16, 8, 4, 2, 1，逐位完成蝶形归约。
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    {
        val += __shfl_xor_sync(mask, val, offset);
    }

    return val;
}

/**
 * @brief 全局求和归约内核（共享内存版本）
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
__global__ void sum_reduce_kernel(const T *__restrict__ input, T *__restrict__ output, size_t n)
{
    extern __shared__ char shared_mem[];
    T *sdata = reinterpret_cast<T *>(shared_mem);

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // 归约求和
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写入结果
    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

/**
 * @brief 全局求和归约内核（warp 洗牌版本）
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
__global__ void sum_reduce_shuffle_kernel(const T *__restrict__ input, T *__restrict__ output, size_t n)
{
    T sum = 0;

    // grid-stride loop，保证每个元素只被访问一次
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        sum += input[idx];
    }

    // 先在 warp 内使用洗牌指令做规约
    sum = warp_shuffle_reduce_sum(sum);

    // 将每个 warp 的部分和写入 shared memory
    __shared__ T warp_sums[32];  // 支持最多 32 个 warp（即 1024 线程的 block）
    int lane   = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (lane == 0)
    {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // 使用第 0 个 warp 对所有 warp 的部分和做一次规约
    T block_sum = 0;
    if (warpId == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum     = (lane < num_warps) ? warp_sums[lane] : static_cast<T>(0);
        block_sum     = warp_shuffle_reduce_sum(block_sum);
    }

    // 线程 0 写回当前 block 的规约结果
    if (threadIdx.x == 0)
    {
        output[blockIdx.x] = block_sum;
    }
}

/**
 * @brief 全局求和归约内核（warp 洗牌 + atomicAdd 版本）
 * @tparam T 数据类型（需支持 CUDA 原生 atomicAdd）
 * @param input 输入数据
 * @param output 单个标量输出
 * @param n 元素总数
 */
template <typename T>
__global__ void sum_reduce_atomic_kernel(const T *__restrict__ input, T *__restrict__ output, size_t n)
{
    T sum = 0;

    // grid-stride loop，遍历所有元素
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += blockDim.x * gridDim.x)
    {
        sum += input[idx];
    }

    // 先在 warp 内使用洗牌指令做规约
    sum = warp_shuffle_reduce_sum(sum);

    // 将每个 warp 的部分和写入 shared memory
    __shared__ T warp_sums[32];  // 支持最多 32 个 warp（即 1024 线程的 block）
    int lane   = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    if (lane == 0)
    {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // 使用第 0 个 warp 对所有 warp 的部分和做一次规约
    T block_sum = 0;
    if (warpId == 0)
    {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        block_sum     = (lane < num_warps) ? warp_sums[lane] : static_cast<T>(0);
        block_sum     = warp_shuffle_reduce_sum(block_sum);
    }

    // 使用 atomicAdd 将当前 block 的规约结果累加到全局输出
    if (threadIdx.x == 0)
    {
        atomicAdd(output, block_sum);
    }
}

/**
 * @brief 沿轴求和内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param outer_size 外层维度大小
 * @param inner_size 内层维度大小
 * @param axis_size 求和轴大小
 */
template <typename T>
__global__ void sum_axis_kernel(const T *__restrict__ input,
                                T *__restrict__ output,
                                size_t outer_size,
                                size_t inner_size,
                                size_t axis_size)
{
    // 核函数启动时，创建的线程块数量为 outer_size，每个线程块内的线程数量为 inner_size
    // 对任意一个高维矩阵都可以想象成是处理3维的矩阵 {outer_size, axis_size, inner_size}
    size_t outer_idx = blockIdx.x;
    size_t inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size)
    {
        T sum           = 0;
        size_t base_idx = outer_idx * axis_size * inner_size + inner_idx;

        for (size_t axis_idx = 0; axis_idx < axis_size; ++axis_idx)
        {
            sum += input[base_idx + axis_idx * inner_size];
        }

        output[outer_idx * inner_size + inner_idx] = sum;
    }
}

/**
 * @brief 启动全局求和内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
void launch_sum_reduce_kernel(const T *input, T *output, size_t n)
{
    const int block_size = 256;
    const int grid_size  = (n + block_size - 1) / block_size;

    // 对支持 CUDA 原生 atomicAdd 的类型，使用单 kernel + atomicAdd 的高效实现
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, int32_t>)
    {
        sum_reduce_atomic_kernel<T><<<grid_size, block_size>>>(input, output, n);
    }
    else
    {
        // 其他类型沿用原来的多轮规约算法，使用 OriginMat 作为临时 workspace（走统一内存池）
        auto workspace_mat =
            OriginMat::zeros(Shape({static_cast<size_t>(grid_size)}), dtype(DataTypeTraits<T>::type).device(kCUDA, 0));
        auto *workspace   = static_cast<OriginMat *>(workspace_mat.get());
        T *temp_output    = workspace->template data_ptr<T>();

        // 第一轮：把 input 分成 grid_size 个 block，每个 block 内部归约，将结果写入 temp_output[blockIdx.x]
        sum_reduce_shuffle_kernel<T><<<grid_size, block_size>>>(input, temp_output, n);

        // 如果有多个块，用 log(n) 的算法进行二分规约。
        if (grid_size > 1)
        {
            size_t remaining = grid_size;

            while (remaining > 1)
            {
                size_t new_grid_size = (remaining + block_size - 1) / block_size;
                sum_reduce_kernel<T>
                    <<<new_grid_size, block_size, block_size * sizeof(T)>>>(temp_output, temp_output, remaining);
                remaining = new_grid_size;
            }
        }

        // 此时 temp_output[0] 存放最终的求和结果，拷贝到输出标量
        CUDA_CHECK(cudaMemcpy(output, temp_output, sizeof(T), cudaMemcpyDeviceToDevice));
    }
}

/**
 * @brief 启动沿轴求和内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param outer_size 外层维度大小
 * @param inner_size 内层维度大小
 * @param axis_size 求和轴大小
 */
template <typename T>
void launch_sum_axis_kernel(const T *input, T *output, size_t outer_size, size_t inner_size, size_t axis_size)
{
    dim3 block(inner_size);
    dim3 grid(outer_size);

    sum_axis_kernel<T><<<grid, block>>>(input, output, outer_size, inner_size, axis_size);
}

/**
 * @brief CUDA求和算子实现
 * @param mat 输入矩阵
 * @param axis 求和轴，-1表示所有元素求和
 * @param keepdim 是否保持维度，默认为false
 * @return 求和结果矩阵
 */
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis, bool keepdim)
{
    // 验证设备类型
    VALIDATE_CUDA_DEVICE(mat);

    const auto &shape = mat.shape();

    if (axis == -1)
    {
        // 全局求和
        Shape result_shape;
        if (keepdim)
        {
            // keepdim=true时保持所有维度为1
            result_shape = Shape(std::vector<size_t>(shape.size(), 1));
        }
        else
        {
            result_shape = Shape({1});
        }

        auto result_mat = OriginMat::zeros(result_shape, dtype(mat.dtype()).device(mat.device()));
        auto *result    = static_cast<OriginMat *>(result_mat.get());

        // 使用类型分发器执行全局求和
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_sum_reduce_kernel(mat.data_ptr<T>(), result->template data_ptr<T>(), mat.elements());
        });

        return result_mat;
    }
    else
    {
        // 沿轴求和
        if (unlikely(axis < 0 || axis >= static_cast<int>(shape.size())))
        {
            THROW_INVALID_ARG("Invalid axis {} for tensor of shape {}", axis, shape.to_string());
        }

        // 计算输出形状
        std::vector<size_t> output_dims;
        if (keepdim)
        {
            // keepdim=true时，在axis位置插入1
            for (size_t i = 0; i < shape.size(); ++i)
            {
                if (i == static_cast<size_t>(axis))
                {
                    output_dims.push_back(1);
                }
                else
                {
                    output_dims.push_back(shape[i]);
                }
            }
        }
        else
        {
            // keepdim=false时，移除指定轴
            output_dims = shape.dims();
            output_dims.erase(output_dims.begin() + axis);
        }
        Shape output_shape(output_dims);

        auto result = std::make_unique<OriginMat>(output_shape, mat.dtype(), mat.device());

        // outer_size：axis 之前所有维度的连乘，表示需要处理的独立外层块数量
        // inner_size：axis 之后所有维度的连乘，表示每个块内需要并行处理的元素数量
        size_t outer_size = 1;
        for (int i = 0; i < axis; ++i)
        {
            outer_size *= shape[i];
        }

        size_t axis_size = shape[axis];

        size_t inner_size = 1;
        for (int i = axis + 1; i < static_cast<int>(shape.size()); ++i)
        {
            inner_size *= shape[i];
        }

        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_sum_axis_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), outer_size, inner_size, axis_size);
        });

        return result;
    }
}

}  // namespace cuda
}  // namespace origin
