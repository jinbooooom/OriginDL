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
 * @brief 维度感知广播CUDA内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param src_strides 输入 strides
 * @param dst_strides 输出 strides
 * @param src_shape 输入形状
 * @param dst_shape 输出形状
 * @param src_ndim 输入维度数
 * @param dst_ndim 输出维度数
 * @param dst_elements 输出元素总数
 */
template <typename T>
__global__ void broadcast_kernel(const T *__restrict__ input,
                                 T *__restrict__ output,
                                 const int *src_strides,
                                 const int *dst_strides,
                                 const int *src_shape,
                                 const int *dst_shape,
                                 int src_ndim,
                                 int dst_ndim,
                                 size_t dst_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dst_elements)
    {
        // 将线性索引转换为多维索引
        int dst_indices[8];  // 假设最大8维
        int temp = idx;
        for (int i = dst_ndim - 1; i >= 0; --i)
        {
            dst_indices[i] = temp % dst_shape[i];
            temp /= dst_shape[i];
        }

        // 计算对应的源索引
        // 从右到左对齐维度（NumPy/PyTorch 广播规则）
        int src_idx = 0;
        for (int i = 0; i < dst_ndim; ++i)
        {
            // 从右到左对齐：dst 的最后一个维度对应 src 的最后一个维度
            int dst_dim_idx = dst_ndim - 1 - i;
            int src_dim_idx = src_ndim - 1 - i;  // 从右到左对齐
            
            if (src_dim_idx >= 0 && src_dim_idx < src_ndim)
            {
                // 如果源维度大小为1，则索引为0（广播）
                int src_dim_size = src_shape[src_dim_idx];
                int src_index = (src_dim_size == 1) ? 0 : dst_indices[dst_dim_idx];
                src_idx += src_index * src_strides[src_dim_idx];
            }
        }

        output[idx] = input[src_idx];
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
    VALIDATE_CUDA_DEVICE(mat);

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

    // 计算 strides
    std::vector<int> src_strides(input_shape.size());
    std::vector<int> dst_strides(target_shape.size());
    std::vector<int> src_shape_vec(input_shape.size());
    std::vector<int> dst_shape_vec(target_shape.size());
    
    int src_stride = 1;
    int dst_stride = 1;
    for (int i = static_cast<int>(input_shape.size()) - 1; i >= 0; --i)
    {
        src_strides[i] = src_stride;
        src_stride *= static_cast<int>(input_shape[i]);
        src_shape_vec[i] = static_cast<int>(input_shape[i]);
    }
    for (int i = static_cast<int>(target_shape.size()) - 1; i >= 0; --i)
    {
        dst_strides[i] = dst_stride;
        dst_stride *= static_cast<int>(target_shape[i]);
        dst_shape_vec[i] = static_cast<int>(target_shape[i]);
    }

    // 将 strides 和 shapes 复制到 GPU
    int *d_src_strides, *d_dst_strides, *d_src_shape, *d_dst_shape;
    size_t src_strides_size = src_strides.size() * sizeof(int);
    size_t dst_strides_size = dst_strides.size() * sizeof(int);
    size_t src_shape_size = src_shape_vec.size() * sizeof(int);
    size_t dst_shape_size = dst_shape_vec.size() * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_src_strides, src_strides_size));
    CUDA_CHECK(cudaMalloc(&d_dst_strides, dst_strides_size));
    CUDA_CHECK(cudaMalloc(&d_src_shape, src_shape_size));
    CUDA_CHECK(cudaMalloc(&d_dst_shape, dst_shape_size));
    
    CUDA_CHECK(cudaMemcpy(d_src_strides, src_strides.data(), src_strides_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst_strides, dst_strides.data(), dst_strides_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_src_shape, src_shape_vec.data(), src_shape_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dst_shape, dst_shape_vec.data(), dst_shape_size, cudaMemcpyHostToDevice));

    // 使用类型分发器执行维度感知广播
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        const int block_size = 256;
        const int grid_size = (target_shape.elements() + block_size - 1) / block_size;
        broadcast_kernel<T><<<grid_size, block_size>>>(
            mat.data_ptr<T>(), result->data_ptr<T>(),
            d_src_strides, d_dst_strides,
            d_src_shape, d_dst_shape,
            static_cast<int>(input_shape.size()),
            static_cast<int>(target_shape.size()),
            target_shape.elements()
        );
    });

    CUDA_CHECK_ASYNC();
    
    // 清理 GPU 内存
    CUDA_CHECK(cudaFree(d_src_strides));
    CUDA_CHECK(cudaFree(d_dst_strides));
    CUDA_CHECK(cudaFree(d_src_shape));
    CUDA_CHECK(cudaFree(d_dst_shape));
    
    return result;
}

}  // namespace cuda
}  // namespace origin
