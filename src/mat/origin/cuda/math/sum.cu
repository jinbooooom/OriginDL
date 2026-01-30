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

    // 把 input 分成 grid_size 个 block
    // 每个 block 内部归约，将结果写入 output[blockIdx.x]，
    // 第一轮归约后，output[0] 存储的是所有 block 的第一个元素的和，output[1] 存储的是所有 block 的第二个元素的和，以此类推
    sum_reduce_kernel<T><<<grid_size, block_size, block_size * sizeof(T)>>>(input, output, n);

    // 如果有多个块，用 log(n) 的算法进行二分规约。
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
        auto result = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());

        // 使用类型分发器执行全局求和
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_sum_reduce_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), mat.elements());
        });

        return result;
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
