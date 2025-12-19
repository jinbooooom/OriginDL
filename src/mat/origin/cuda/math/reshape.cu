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
 * @brief CUDA重塑算子实现
 * @details 重新排列张量的形状，但不改变数据内容
 *
 * ============================================================================
 * PyTorch reshape行为详解
 * ============================================================================
 *
 * PyTorch的reshape行为取决于张量的内存连续性：
 *
 * 1. 连续张量（contiguous tensor）的reshape：
 *    - 如果张量本身是连续的，reshape只是改变视图，不会拷贝数据
 *    - 返回的新张量与原始张量共享相同的内存存储
 *    - 这是零拷贝操作，性能很高
 *
 *    示例：
 *    ```python
 *    import torch
 *    a = torch.arange(12)  # 连续张量
 *    b = a.reshape(3, 4)   # 只是视图，不拷贝数据
 *    print(b.storage().data_ptr() == a.storage().data_ptr())  # True
 *    ```
 *
 * 2. 非连续张量（non-contiguous tensor）的reshape：
 *    - 如果张量本身不是连续的（如经过transpose、permute等操作后），
 *      reshape会创建新的数据拷贝
 *    - 数据在内存中的顺序会被重新排列
 *    - 性能开销较大，但确保数据连续性
 *
 *    示例：
 *    ```python
 *    a = torch.arange(12).reshape(3, 4)
 *    b = a.t()  # 转置后，b不连续
 *    print(b.is_contiguous())  # False
 *    c = b.reshape(12)  # 这里会拷贝数据
 *    print(c.storage().data_ptr() == b.storage().data_ptr())  # False
 *    ```
 *
 * 3. 与view()的区别：
 *    - view()：只能用于连续的张量，始终返回视图
 *    - reshape()：可以处理非连续的张量，必要时会进行数据复制
 *
 * ============================================================================
 * 当前实现说明
 * ============================================================================
 *
 * 当前实现采用基于拷贝的reshape策略：
 * - 与CPU版本保持一致，创建新的存储并复制数据
 * - 简单可靠，但性能不是最优
 * - 适用于所有情况，包括非连续张量
 *
 * ============================================================================
 * 未来优化计划
 * ============================================================================
 *
 * 计划实现类似PyTorch的智能reshape行为：
 * 1. 检查张量是否连续（contiguous）
 * 2. 如果连续，返回视图（共享存储，零拷贝）
 * 3. 如果不连续，创建新存储并复制数据
 *
 * 实现步骤：
 * - 添加is_contiguous()方法检查张量连续性
 * - 修改reshape逻辑，根据连续性选择视图或拷贝策略
 * - 保持与PyTorch API的兼容性
 */

/**
 * @brief CUDA reshape算子实现
 * @param mat 输入矩阵
 * @param new_shape 新的形状
 * @return 重塑后的矩阵
 */
std::unique_ptr<Mat> reshape(const OriginMat &mat, const Shape &new_shape)
{
    // 验证输入
    if (new_shape.elements() != mat.elements())
    {
        THROW_INVALID_ARG("Reshape: total elements must match. Original: {}, Target: {}", mat.elements(),
                          new_shape.elements());
    }

    // 验证设备类型
    VALIDATE_CUDA_DEVICE(mat);

    // 创建新的OriginMat，使用新的形状
    auto result = std::make_unique<OriginMat>(new_shape, mat.dtype(), mat.device());

    // 使用CUDA内存复制
    cudaError_t err = cudaMemcpy(result->storage()->data(), mat.storage()->data(),
                                 mat.elements() * get_type_size(mat.dtype()), cudaMemcpyDeviceToDevice);

    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA memory copy failed in reshape: {}", cudaGetErrorString(err));
    }

    CUDA_CHECK_ASYNC();

    return result;
}

}  // namespace cuda
}  // namespace origin
