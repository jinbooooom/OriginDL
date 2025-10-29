#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA取负算子实现
 * @details 计算输入张量每个元素的负值
 */

/**
 * @brief negate算子实现
 * @param mat 输入矩阵
 * @return 取负运算结果矩阵
 */
std::unique_ptr<Mat> negate(const OriginMat &mat)
{
    // 验证输入
    if (unlikely(mat.device().type() != DeviceType::kCUDA))
    {
        THROW_INVALID_ARG("Device mismatch in CUDA negate: expected CUDA device, got {}", mat.device().to_string());
    }

    // 创建结果张量
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 直接调用一元内核
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_unary_kernel<T, NegOp>(static_cast<const T *>(a_data), 
                                     static_cast<T *>(c_data), 
                                     mat.elements(), NegOp{}, 0);
    });

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
