#include <algorithm>
#include <limits>
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief 全局最大值归约内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param n 元素总数
 */
template <typename T>
__global__ void max_reduce_kernel(const T *__restrict__ input, T *__restrict__ output, size_t n)
{
    extern __shared__ char shared_mem[];
    T *sdata = reinterpret_cast<T *>(shared_mem);

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    // 加载数据到共享内存
    sdata[tid] = (i < n) ? input[i] : std::numeric_limits<T>::lowest();
    __syncthreads();

    // 归约求最大值
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
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
 * @brief 沿轴最大值内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param outer_size 外层维度大小
 * @param inner_size 内层维度大小
 * @param axis_size 最大值轴大小
 */
template <typename T>
__global__ void max_axis_kernel(const T *__restrict__ input,
                                 T *__restrict__ output,
                                 size_t outer_size,
                                 size_t inner_size,
                                 size_t axis_size)
{
    size_t outer_idx = blockIdx.x;
    size_t inner_idx = threadIdx.x;

    if (outer_idx < outer_size && inner_idx < inner_size)
    {
        T max_val      = std::numeric_limits<T>::lowest();
        size_t base_idx = outer_idx * axis_size * inner_size + inner_idx;

        for (size_t axis_idx = 0; axis_idx < axis_size; ++axis_idx)
        {
            max_val = max(max_val, input[base_idx + axis_idx * inner_size]);
        }

        output[outer_idx * inner_size + inner_idx] = max_val;
    }
}

/**
 * @brief 启动沿轴最大值内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param outer_size 外层维度大小
 * @param inner_size 内层维度大小
 * @param axis_size 最大值轴大小
 */
template <typename T>
void launch_max_axis_kernel(const T *input, T *output, size_t outer_size, size_t inner_size, size_t axis_size)
{
    dim3 block(inner_size);
    dim3 grid(outer_size);
    max_axis_kernel<T><<<grid, block>>>(input, output, outer_size, inner_size, axis_size);
}

/**
 * @brief CUDA max：沿指定轴计算最大值
 * @param mat 输入矩阵
 * @param axis 计算轴，-1 表示所有元素
 * @return 最大值结果矩阵
 */
std::unique_ptr<Mat> max(const OriginMat &mat, int axis)
{
    VALIDATE_SAME_CUDA_DEVICE(mat, mat);

    if (axis == -1)
    {
        // 对所有元素求最大值，返回标量
        auto result_shape = Shape{1};  // 标量结果
        auto result       = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());

        size_t n = mat.elements();
        if (n == 0)
        {
            THROW_INVALID_ARG("max: cannot compute max of empty tensor");
        }

        // 使用归约内核
        const size_t threads_per_block = 256;
        const size_t num_blocks        = (n + threads_per_block - 1) / threads_per_block;

        // 使用递归归约（参考 sum 的实现）
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            T *output_ptr = result->data_ptr<T>();
            
            // 第一轮归约
            max_reduce_kernel<T><<<num_blocks, threads_per_block, threads_per_block * sizeof(T)>>>(
                mat.data_ptr<T>(), output_ptr, n);
            
            CUDA_CHECK_ASYNC();
            
            // 如果还有多个块，进行多轮归约
            if (num_blocks > 1)
            {
                size_t remaining = num_blocks;
                while (remaining > 1)
                {
                    size_t new_num_blocks = (remaining + threads_per_block - 1) / threads_per_block;
                    max_reduce_kernel<T><<<new_num_blocks, threads_per_block, threads_per_block * sizeof(T)>>>(
                        output_ptr, output_ptr, remaining);
                    CUDA_CHECK_ASYNC();
                    remaining = new_num_blocks;
                }
            }
        });

        CUDA_CHECK_ASYNC();
        return result;
    }

    // 验证轴的有效性
    if (axis < 0 || axis >= static_cast<int>(mat.shape().size()))
    {
        THROW_INVALID_ARG("Invalid axis {} for max operation. Tensor has {} dimensions", axis, mat.shape().size());
    }

    // 计算结果形状：移除指定轴
    std::vector<size_t> result_dims;
    for (size_t i = 0; i < mat.shape().size(); ++i)
    {
        if (i != static_cast<size_t>(axis))
        {
            result_dims.push_back(mat.shape()[i]);
        }
    }
    Shape result_shape(result_dims);

    auto result = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());

    // 计算维度大小
    size_t outer_size = 1;
    for (size_t i = 0; i < static_cast<size_t>(axis); ++i)
    {
        outer_size *= mat.shape()[i];
    }

    size_t axis_size = mat.shape()[axis];

    size_t inner_size = 1;
    for (size_t i = axis + 1; i < mat.shape().size(); ++i)
    {
        inner_size *= mat.shape()[i];
    }

    // 启动内核（参考 sum 的实现）
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_max_axis_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), outer_size, inner_size, axis_size);
    });

    CUDA_CHECK_ASYNC();
    return result;
}

}  // namespace cuda
}  // namespace origin
