#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA广播算子实现
 * @details 实现高效的GPU广播运算
 *
 * ============================================================================
 * PyTorch broadcast_to行为详解
 * ============================================================================
 *
 * PyTorch的broadcast_to算子支持多种广播场景：
 *
 * 1. 维度扩展：
 *    - 在张量前面添加维度
 *    - 新维度大小为1，可以广播到任意大小
 *
 *    示例：
 *    ```python
 *    import torch
 *    x = torch.tensor([1, 2, 3])  # shape: (3,)
 *    result = torch.broadcast_to(x, (2, 3))  # shape: (2, 3)
 *    # 结果: [[1, 2, 3], [1, 2, 3]]
 *    ```
 *
 * 2. 维度放大：
 *    - 将大小为1的维度放大到指定大小
 *    - 数据沿该维度重复
 *
 *    示例：
 *    ```python
 *    x = torch.tensor([[1, 2, 3]])  # shape: (1, 3)
 *    result = torch.broadcast_to(x, (4, 3))  # shape: (4, 3)
 *    # 结果: [[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]]
 *    ```
 *
 * 3. 复杂广播：
 *    - 支持多个维度的同时广播
 *    - 遵循NumPy广播规则
 *
 *    示例：
 *    ```python
 *    x = torch.tensor([[[1, 2, 3]]])  # shape: (1, 1, 3)
 *    result = torch.broadcast_to(x, (2, 4, 3))  # shape: (2, 4, 3)
 *    ```
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于CUDA内核的广播策略：
 * - 实现高效的广播内核
 * - 支持所有数值类型
 * - 使用优化的内存访问模式
 * - 与CPU版本保持一致的行为
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现更高级的优化：
 * 1. 使用更高效的广播算法
 * 2. 支持更复杂的广播模式
 * 3. 实现零拷贝广播（视图）
 * 4. 优化内存访问模式
 *
 * 实现步骤：
 * - 实现零拷贝广播
 * - 优化内核性能
 * - 添加更多广播测试
 * - 保持与PyTorch API的兼容性
 */

/**
 * @brief 广播CUDA内核（简化版本，参考CPU实现）
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param src_elements 输入元素总数
 * @param dst_elements 输出元素总数
 */
template <typename T>
__global__ void broadcast_kernel(const T *__restrict__ input,
                                 T *__restrict__ output,
                                 size_t src_elements,
                                 size_t dst_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dst_elements)
    {
        // 使用模运算实现循环复制，与CPU版本保持一致
        output[idx] = input[idx % src_elements];
    }
}

/**
 * @brief 启动广播内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param src_elements 输入元素总数
 * @param dst_elements 输出元素总数
 */
template <typename T>
void launch_broadcast_kernel(const T *input, T *output, size_t src_elements, size_t dst_elements)
{
    const int block_size = 256;
    const int grid_size  = (dst_elements + block_size - 1) / block_size;

    broadcast_kernel<T><<<grid_size, block_size>>>(input, output, src_elements, dst_elements);

    // 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA broadcast kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 验证广播兼容性
 * @param input_shape 输入形状
 * @param output_shape 输出形状
 */
void validate_broadcast_compatibility(const Shape &input_shape, const Shape &output_shape)
{
    if (input_shape.size() > output_shape.size())
    {
        THROW_INVALID_ARG("Input shape {} has more dimensions than output shape {}", input_shape.to_string(),
                          output_shape.to_string());
    }

    int input_ndim  = input_shape.size();
    int output_ndim = output_shape.size();

    // 从右到左检查维度兼容性
    for (int i = 0; i < input_ndim; ++i)
    {
        int input_idx  = input_ndim - 1 - i;
        int output_idx = output_ndim - 1 - i;

        size_t input_dim  = input_shape[input_idx];
        size_t output_dim = output_shape[output_idx];

        if (input_dim != 1 && input_dim != output_dim)
        {
            THROW_INVALID_ARG("Cannot broadcast from {} to {}: dimension {} mismatch", input_shape.to_string(),
                              output_shape.to_string(), i);
        }
    }
}

/**
 * @brief CUDA广播算子实现
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return 广播后的矩阵
 */
std::unique_ptr<Mat> broadcast_to(const OriginMat &mat, const Shape &target_shape)
{
    // 验证设备类型
    validation::validate_cuda_device(mat, "broadcast_to");

    const auto &input_shape = mat.shape();

    // 验证广播兼容性
    validate_broadcast_compatibility(input_shape, target_shape);

    // 如果形状相同，直接返回副本
    if (input_shape == target_shape)
    {
        auto result     = std::make_unique<OriginMat>(target_shape, mat.dtype(), mat.device());
        cudaError_t err = cudaMemcpy(result->storage()->data(), mat.storage()->data(),
                                     mat.elements() * get_type_size(mat.dtype()), cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
            THROW_RUNTIME_ERROR("CUDA memory copy failed in broadcast_to: {}", cudaGetErrorString(err));
        }
        return result;
    }

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(target_shape, mat.dtype(), mat.device());

    // 使用类型分发器执行广播（简化版本，参考CPU实现）
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_broadcast_kernel(mat.data_ptr<T>(), result->data_ptr<T>(), mat.elements(), target_shape.elements());
    });

    cudaDeviceSynchronize();
    return result;
}

}  // namespace cuda
}  // namespace origin
