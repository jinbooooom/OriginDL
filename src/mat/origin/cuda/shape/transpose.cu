#include <vector>
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
 * @brief CUDA转置算子实现
 * @details 重新排列张量的维度，交换最后两个维度
 *
 * ============================================================================
 * PyTorch transpose行为详解
 * ============================================================================
 *
 * PyTorch的transpose行为采用视图转置策略：
 *
 * 1. 视图转置（View Transpose）：
 *    - 只改变张量的形状和步长（stride），不重新排列内存中的数据
 *    - 零拷贝操作，性能高
 *    - 数据在内存中的顺序保持不变
 *    - 通过改变索引计算方式来"模拟"转置效果
 *    - 适用于大多数情况，特别是深度学习框架中的转置操作
 *
 *    示例：
 *    ```python
 *    import torch
 *    a = torch.arange(12).reshape(3, 4)
 *    print("Original data:", a.flatten())
 *    # Output: tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])
 *
 *    b = a.t()  # 转置
 *    print("Transposed data:", b.flatten())
 *    # Output: tensor([ 0,  4,  8,  1,  5,  9,  2,  6, 10,  3,  7, 11])
 *
 *    print("Same storage:", a.storage().data_ptr() == b.storage().data_ptr())
 *    # Output: True (共享存储)
 *    ```
 *
 * 2. 数据转置（Data Transpose）：
 *    - 真正重新排列内存中的数据
 *    - 需要分配新内存并复制数据
 *    - 数据在内存中的顺序被重新排列
 *    - 性能开销较大
 *    - 主要用于矩阵乘法等需要真正转置数据的操作
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于拷贝的数据转置策略：
 * - 与CPU版本保持一致，创建新的存储并重新排列数据
 * - 真正的数据转置，数据在内存中的顺序被重新排列
 * - 简单可靠，但性能不是最优
 * - 适用于所有情况，包括需要真正转置数据的操作
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现类似PyTorch的视图转置行为：
 * 1. 实现视图转置，只改变步长信息
 * 2. 提供数据转置选项，用于需要真正转置数据的场景
 * 3. 根据使用场景自动选择最优策略
 *
 * 实现步骤：
 * - 添加视图转置实现，修改步长信息
 * - 保留数据转置作为选项
 * - 提供API让用户选择转置策略
 * - 保持与PyTorch API的兼容性
 */

/**
 * @brief 2D矩阵转置CUDA内核
 * @tparam T 数据类型
 * @param input 输入矩阵数据
 * @param output 输出矩阵数据
 * @param rows 行数
 * @param cols 列数
 */
template <typename T>
__global__ void transpose_2d_kernel(const T *__restrict__ input, T *__restrict__ output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        // 转置：output[col][row] = input[row][col]
        output[col * rows + row] = input[row * cols + col];
    }
}

/**
 * @brief 高维张量转置CUDA内核（转置最后两个维度）
 * @tparam T 数据类型
 * @param input 输入张量数据
 * @param output 输出张量数据
 * @param last_dim 最后一个维度大小
 * @param second_last_dim 倒数第二个维度大小
 * @param outer_elements 外层元素数量
 */
template <typename T>
__global__ void transpose_nd_kernel(const T *__restrict__ input,
                                    T *__restrict__ output,
                                    int last_dim,
                                    int second_last_dim,
                                    int outer_elements)
{
    int idx            = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = outer_elements * last_dim * second_last_dim;

    if (idx < total_elements)
    {
        // 计算当前元素在张量中的位置
        int outer     = idx / (last_dim * second_last_dim);
        int remaining = idx % (last_dim * second_last_dim);
        int j         = remaining / second_last_dim;  // 原最后一个维度的索引
        int i         = remaining % second_last_dim;  // 原倒数第二个维度的索引

        // 计算源索引和目标索引
        int src_idx = outer * (last_dim * second_last_dim) + i * last_dim + j;
        int dst_idx = outer * (last_dim * second_last_dim) + j * second_last_dim + i;

        output[dst_idx] = input[src_idx];
    }
}

/**
 * @brief 启动2D转置内核
 * @tparam T 数据类型
 * @param input 输入矩阵数据
 * @param output 输出矩阵数据
 * @param rows 行数
 * @param cols 列数
 */
template <typename T>
void launch_transpose_2d_kernel(const T *input, T *output, int rows, int cols)
{
    // 设置线程块大小
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    // 启动内核
    transpose_2d_kernel<T><<<grid, block>>>(input, output, rows, cols);
}

/**
 * @brief 启动高维转置内核
 * @tparam T 数据类型
 * @param input 输入张量数据
 * @param output 输出张量数据
 * @param last_dim 最后一个维度大小
 * @param second_last_dim 倒数第二个维度大小
 * @param outer_elements 外层元素数量
 */
template <typename T>
void launch_transpose_nd_kernel(const T *input, T *output, int last_dim, int second_last_dim, int outer_elements)
{
    int total_elements = outer_elements * last_dim * second_last_dim;

    // 设置线程块大小
    dim3 block(256);
    dim3 grid((total_elements + block.x - 1) / block.x);

    // 启动内核
    transpose_nd_kernel<T><<<grid, block>>>(input, output, last_dim, second_last_dim, outer_elements);
}

/**
 * @brief CUDA data_transpose算子实现（数据转置）
 * @param mat 输入矩阵
 * @return 转置后的矩阵
 *
 * @details 使用数据转置策略：真正重新排列内存中的数据
 *
 * 数据转置的特点：
 * - 需要分配新内存并复制数据
 * - 数据在内存中的顺序被重新排列
 * - 转置后的张量是连续的
 * - 适用于需要真正转置数据的操作（如矩阵乘法）
 *
 * 注意：当前实现只转置最后两个维度
 */
std::unique_ptr<Mat> data_transpose(const OriginMat &mat)
{
    // 验证设备类型
    VALIDATE_CUDA_DEVICE(mat);

    if (mat.shape().size() == 0)
    {
        // 0维张量（标量）：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 1)
    {
        // 一维张量：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 2)
    {
        // 二维张量：转置最后两个维度（数据转置，重新排列数据）
        Shape new_shape({mat.shape()[1], mat.shape()[0]});
        auto result = std::make_unique<OriginMat>(new_shape, mat.dtype(), mat.device());

        // 使用类型分发器执行转置操作
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_transpose_2d_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), static_cast<int>(mat.shape()[0]),
                                       static_cast<int>(mat.shape()[1]));
        });

        CUDA_CHECK_ASYNC();

        return result;
    }
    else
    {
        // 高维张量：转置最后两个维度
        std::vector<size_t> new_dims = mat.shape().dims();
        std::swap(new_dims[new_dims.size() - 2], new_dims[new_dims.size() - 1]);
        Shape new_shape(new_dims);

        auto result = std::make_unique<OriginMat>(new_shape, mat.dtype(), mat.device());

        // 计算转置参数
        const size_t last_dim        = mat.shape()[mat.shape().size() - 1];
        const size_t second_last_dim = mat.shape()[mat.shape().size() - 2];
        const size_t outer_elements  = mat.elements() / (last_dim * second_last_dim);

        // 使用类型分发器执行转置操作
        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            launch_transpose_nd_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), static_cast<int>(last_dim),
                                       static_cast<int>(second_last_dim), static_cast<int>(outer_elements));
        });

        CUDA_CHECK_ASYNC();

        return result;
    }
}

/**
 * @brief CUDA view_transpose算子实现（视图转置）
 * @param mat 输入矩阵
 * @return 转置后的矩阵（视图，共享 storage）
 *
 * @details 使用视图转置策略：只改变 shape 和 strides，不重新排列内存中的数据
 *
 * 视图转置的特点：
 * - 零拷贝操作，性能高
 * - 数据在内存中的顺序保持不变
 * - 通过改变 strides 来"模拟"转置效果
 * - 转置后的张量是非连续的，如果需要进行元素级操作，需要先调用 contiguous()
 *
 * 注意：当前实现只转置最后两个维度
 */
std::unique_ptr<Mat> view_transpose(const OriginMat &mat)
{
    // 验证设备类型
    VALIDATE_CUDA_DEVICE(mat);

    if (mat.shape().size() == 0)
    {
        // 0维张量（标量）：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 1)
    {
        // 一维张量：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else
    {
        // 二维或高维张量：转置最后两个维度（视图转置）
        // 计算新的形状
        std::vector<size_t> new_dims = mat.shape().dims();
        std::swap(new_dims[new_dims.size() - 2], new_dims[new_dims.size() - 1]);
        Shape new_shape(new_dims);

        // 计算新的 strides：交换最后两个维度的 strides
        std::vector<size_t> new_strides = mat.strides();
        std::swap(new_strides[new_strides.size() - 2], new_strides[new_strides.size() - 1]);

        // 使用视图构造函数创建转置后的张量（共享 storage，只改变 shape 和 strides）
        return std::make_unique<OriginMat>(mat.storage(), new_shape, new_strides, mat.dtype());
    }
}

/**
 * @brief CUDA transpose算子实现（默认使用视图转置）
 * @param mat 输入矩阵
 * @return 转置后的矩阵（视图，共享 storage）
 *
 * @details 调用 view_transpose 实现视图转置
 *
 * 视图转置的特点：
 * - 零拷贝操作，性能高
 * - 数据在内存中的顺序保持不变
 * - 通过改变 strides 来"模拟"转置效果
 * - 转置后的张量是非连续的，如果需要进行元素级操作，需要先调用 contiguous()
 */
std::unique_ptr<Mat> transpose(const OriginMat &mat)
{
#if 0
    return view_transpose(mat);
#else
    return data_transpose(mat);
#endif
}

}  // namespace cuda
}  // namespace origin
