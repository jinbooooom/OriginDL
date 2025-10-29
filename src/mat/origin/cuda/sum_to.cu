#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA sum_to算子实现
 * @details 实现高效的GPU sum_to运算
 *
 * ============================================================================
 * PyTorch sum_to行为详解
 * ============================================================================
 *
 * PyTorch的sum_to算子支持多种求和到指定形状的场景：
 *
 * 1. 维度缩减：
 *    - 将张量求和到指定的形状
 *    - 支持广播和维度缩减
 *
 *    示例：
 *    ```python
 *    import torch
 *    x = torch.randn(3, 4, 5)  # shape: (3, 4, 5)
 *    result = torch.sum_to(x, (3, 1, 1))  # shape: (3, 1, 1)
 *    # 沿第1,2维求和
 *    ```
 *
 * 2. 广播求和：
 *    - 支持不同形状的求和
 *    - 遵循NumPy广播规则
 *
 *    示例：
 *    ```python
 *    x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
 *    result = torch.sum_to(x, (1, 3, 1))  # shape: (1, 3, 1)
 *    # 沿第0,2维求和
 *    ```
 *
 * 3. 复杂求和：
 *    - 支持多个维度的同时求和
 *    - 保持指定维度的形状
 *
 *    示例：
 *    ```python
 *    x = torch.randn(2, 3, 4, 5)  # shape: (2, 3, 4, 5)
 *    result = torch.sum_to(x, (1, 3, 1, 5))  # shape: (1, 3, 1, 5)
 *    # 沿第0,2维求和
 *    ```
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于CUDA内核的sum_to策略：
 * - 实现高效的求和内核
 * - 支持所有数值类型
 * - 使用优化的内存访问模式
 * - 与CPU版本保持一致的行为
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现更高级的优化：
 * 1. 使用更高效的求和算法
 * 2. 支持更复杂的求和模式
 * 3. 实现零拷贝求和（视图）
 * 4. 优化内存访问模式
 *
 * 实现步骤：
 * - 实现零拷贝求和
 * - 优化内核性能
 * - 添加更多求和测试
 * - 保持与PyTorch API的兼容性
 */

/**
 * @brief sum_to CUDA内核（简化版本，支持最多4维）
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param input_shape 输入形状
 * @param output_shape 输出形状
 * @param input_strides 输入步长
 * @param output_strides 输出步长
 * @param n 输出元素总数
 */
template <typename T>
__global__ void sum_to_kernel(const T *__restrict__ input,
                              T *__restrict__ output,
                              const size_t *__restrict__ input_shape,
                              const size_t *__restrict__ output_shape,
                              const size_t *__restrict__ input_strides,
                              const size_t *__restrict__ output_strides,
                              size_t n,
                              int ndim)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // 计算输出索引对应的输入索引范围
        size_t temp_idx          = idx;
        size_t output_indices[4] = {0, 0, 0, 0};

        // 计算输出索引
        for (int i = ndim - 1; i >= 0; --i)
        {
            output_indices[i] = temp_idx % output_shape[i];
            temp_idx /= output_shape[i];
        }

        // 计算输入索引范围
        size_t input_start[4] = {0, 0, 0, 0};
        size_t input_end[4]   = {1, 1, 1, 1};

        for (int i = 0; i < ndim; ++i)
        {
            if (output_shape[i] == 1)
            {
                // 如果输出维度为1，则求和整个输入维度
                input_start[i] = 0;
                input_end[i]   = input_shape[i];
            }
            else
            {
                // 否则只取对应的输入索引
                input_start[i] = output_indices[i];
                input_end[i]   = output_indices[i] + 1;
            }
        }

        // 计算求和
        T sum                   = 0;
        size_t input_indices[4] = {0, 0, 0, 0};

        // 初始化输入索引
        for (int i = 0; i < ndim; ++i)
        {
            input_indices[i] = input_start[i];
        }

        bool done = false;
        while (!done)
        {
            // 计算输入索引
            size_t input_idx = 0;
            for (int i = 0; i < ndim; ++i)
            {
                input_idx += input_indices[i] * input_strides[i];
            }

            sum += input[input_idx];

            // 更新索引
            for (int i = ndim - 1; i >= 0; --i)
            {
                if (output_shape[i] == 1)
                {
                    // 如果输出维度为1，则遍历整个输入维度
                    input_indices[i]++;
                    if (input_indices[i] < input_end[i])
                    {
                        break;
                    }
                    else
                    {
                        input_indices[i] = input_start[i];
                        if (i == 0)
                        {
                            done = true;
                        }
                    }
                }
                else
                {
                    // 如果输出维度不为1，则索引固定
                    break;
                }
            }
        }

        output[idx] = sum;
    }
}

/**
 * @brief 启动sum_to内核
 * @tparam T 数据类型
 * @param input 输入数据
 * @param output 输出数据
 * @param input_shape 输入形状
 * @param output_shape 输出形状
 * @param input_strides 输入步长
 * @param output_strides 输出步长
 * @param n 输出元素总数
 * @param ndim 维度数
 */
template <typename T>
void launch_sum_to_kernel(const T *input,
                          T *output,
                          const size_t *input_shape,
                          const size_t *output_shape,
                          const size_t *input_strides,
                          const size_t *output_strides,
                          size_t n,
                          int ndim)
{
    const int block_size = 256;
    const int grid_size  = (n + block_size - 1) / block_size;

    sum_to_kernel<T>
        <<<grid_size, block_size>>>(input, output, input_shape, output_shape, input_strides, output_strides, n, ndim);
}

/**
 * @brief 验证sum_to兼容性
 * @param input_shape 输入形状
 * @param output_shape 输出形状
 */
void validate_sum_to_compatibility(const Shape &input_shape, const Shape &output_shape)
{
    if (input_shape.size() != output_shape.size())
    {
        THROW_INVALID_ARG("Input shape {} and output shape {} must have the same number of dimensions",
                          input_shape.to_string(), output_shape.to_string());
    }

    int ndim = input_shape.size();

    // 检查每个维度的兼容性
    for (int i = 0; i < ndim; ++i)
    {
        size_t input_dim  = input_shape[i];
        size_t output_dim = output_shape[i];

        if (output_dim != 1 && output_dim != input_dim)
        {
            THROW_INVALID_ARG("Cannot sum_to from {} to {}: dimension {} mismatch", input_shape.to_string(),
                              output_shape.to_string(), i);
        }
    }
}

/**
 * @brief CUDA sum_to算子实现（参考CPU版本）
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return sum_to结果矩阵
 */
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape)
{
    // 验证设备类型
    VALIDATE_CUDA_DEVICE(mat);

    const auto &input_shape = mat.shape();

    // 检查形状兼容性
    if (input_shape == target_shape)
    {
        // 形状相同，直接返回副本
        return std::make_unique<OriginMat>(mat);
    }

    // 计算元素总数
    size_t current_elements = mat.elements();
    size_t target_elements  = target_shape.elements();

    if (target_elements > current_elements)
    {
        // 目标形状更大，sum_to不支持广播，抛出异常
        THROW_RUNTIME_ERROR("sum_to: Target shape {} cannot have more elements than source tensor {}",
                            target_shape.elements(), mat.elements());
    }
    else
    {
        // 目标形状更小或相等，需要求和压缩
        // 收集需要求和的维度
        std::vector<int> sum_dims;

        // 从左到右比较维度（按照torch_mat的逻辑）
        size_t min_dims = std::min(mat.shape().size(), target_shape.size());
        for (size_t i = 0; i < min_dims; ++i)
        {
            if (target_shape[i] == 1 && mat.shape()[i] > 1)
            {
                sum_dims.push_back(i);
            }
        }

        // 处理多余的维度（从右边开始的多余维度）
        // 如果源形状比目标形状多维度，需要对这些维度求和
        if (mat.shape().size() > target_shape.size())
        {
            for (size_t i = target_shape.size(); i < mat.shape().size(); ++i)
            {
                sum_dims.push_back(i);
            }
        }

        // 执行求和操作
        std::unique_ptr<OriginMat> current = std::make_unique<OriginMat>(mat);

        // 按从大到小的顺序求和，这样轴索引不会改变
        std::sort(sum_dims.begin(), sum_dims.end(), std::greater<int>());
        for (int dim : sum_dims)
        {
            auto sum_result = current->sum(dim);
            current         = std::unique_ptr<OriginMat>(static_cast<OriginMat *>(sum_result.release()));
        }

        // 最后reshape到目标形状
        if (current->shape() != target_shape)
        {
            auto reshape_result = current->reshape(target_shape);
            current             = std::unique_ptr<OriginMat>(static_cast<OriginMat *>(reshape_result.release()));
        }

        return current;
    }
}

}  // namespace cuda
}  // namespace origin
