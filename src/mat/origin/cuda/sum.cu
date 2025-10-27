#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA求和算子实现
 * @details 实现高效的GPU求和运算
 *
 * ============================================================================
 * PyTorch sum行为详解
 * ============================================================================
 *
 * PyTorch的sum算子支持多种求和场景：
 *
 * 1. 全局求和：
 *    - 对所有元素求和，返回标量
 *    - 使用高效的归约算法
 *
 *    示例：
 *    ```python
 *    import torch
 *    x = torch.randn(3, 4)
 *    result = torch.sum(x)  # 标量结果
 *    ```
 *
 * 2. 沿轴求和：
 *    - 沿指定轴求和，保持其他维度
 *    - 支持负轴索引
 *
 *    示例：
 *    ```python
 *    x = torch.randn(3, 4, 5)
 *    result = torch.sum(x, dim=1)  # 沿第1维求和，结果形状: (3, 5)
 *    result = torch.sum(x, dim=-1)  # 沿最后一维求和，结果形状: (3, 4)
 *    ```
 *
 * 3. 多轴求和：
 *    - 同时沿多个轴求和
 *    - 支持keepdim参数保持维度
 *
 *    示例：
 *    ```python
 *    x = torch.randn(3, 4, 5)
 *    result = torch.sum(x, dim=(1, 2))  # 沿第1,2维求和，结果形状: (3,)
 *    result = torch.sum(x, dim=(1, 2), keepdim=True)  # 结果形状: (3, 1, 1)
 *    ```
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于CUDA内核的求和策略：
 * - 实现高效的归约求和内核
 * - 支持全局求和和沿轴求和
 * - 使用优化的内存访问模式
 * - 与CPU版本保持一致的行为
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现更高级的优化：
 * 1. 使用cuBLAS库进行高性能归约
 * 2. 实现更复杂的多轴求和
 * 3. 支持keepdim参数
 * 4. 实现更高效的归约算法
 *
 * 实现步骤：
 * - 集成cuBLAS库调用
 * - 实现多轴求和逻辑
 * - 添加keepdim支持
 * - 保持与PyTorch API的兼容性
 */

/**
 * @brief 全局求和归约内核
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

    // 第一轮归约
    sum_reduce_kernel<T><<<grid_size, block_size, block_size * sizeof(T)>>>(input, output, n);

    // 如果还有多个块，进行第二轮归约
    if (grid_size > 1)
    {
        T *temp_output   = output;
        size_t remaining = grid_size;

        while (remaining > 1)
        {
            size_t new_grid_size = (remaining + block_size - 1) / block_size;
            sum_reduce_kernel<T>
                <<<new_grid_size, block_size, block_size * sizeof(T)>>>(temp_output, temp_output, remaining);
            remaining = new_grid_size;
        }
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
 * @return 求和结果矩阵
 */
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis)
{
    // 验证设备类型
    validation::validate_cuda_device(mat, "sum");

    const auto &shape = mat.shape();

    if (axis == -1)
    {
        // 全局求和
        auto result = std::make_unique<OriginMat>(Shape({1}), mat.dtype(), mat.device());

        // 使用类型分发器执行全局求和
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_sum_reduce_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), mat.elements());
        });

        cudaDeviceSynchronize();
        return result;
    }
    else
    {
        // 沿轴求和
        if (axis < 0 || axis >= static_cast<int>(shape.size()))
        {
            THROW_INVALID_ARG("Invalid axis {} for tensor of shape {}", axis, shape.to_string());
        }

        // 计算输出形状
        std::vector<size_t> output_dims = shape.dims();
        output_dims.erase(output_dims.begin() + axis);
        Shape output_shape(output_dims);

        auto result = std::make_unique<OriginMat>(output_shape, mat.dtype(), mat.device());

        // 计算维度信息
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

        // 使用类型分发器执行沿轴求和
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_sum_axis_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), outer_size, inner_size, axis_size);
        });

        cudaDeviceSynchronize();
        return result;
    }
}

}  // namespace cuda
}  // namespace origin
