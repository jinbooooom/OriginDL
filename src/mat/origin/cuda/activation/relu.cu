#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA ReLU 激活函数实现
 * @param mat 输入矩阵
 * @return ReLU 运算结果矩阵，y = max(0, x)
 */
std::unique_ptr<Mat> relu(const OriginMat &mat)
{
    // 输入验证
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute ReLU of empty matrix");
    }
    VALIDATE_CUDA_DEVICE(mat);

    // 创建结果矩阵
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 使用类型分发器执行 ReLU 运算
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_unary_kernel<T, ReLUOp>(static_cast<const T *>(a_data), static_cast<T *>(c_data), mat.elements(),
                                       ReLUOp{}, 0);
    });

    CUDA_CHECK_ASYNC();

    return result;
}

/**
 * @brief CUDA原地ReLU激活函数实现（修改当前矩阵）
 * @param mat 输入矩阵（会被修改）
 */
void relu_inplace(OriginMat &mat)
{
    // 输入验证
    if (unlikely(mat.elements() == 0))
    {
        THROW_INVALID_ARG("Cannot compute ReLU of empty matrix");
    }
    VALIDATE_CUDA_DEVICE(mat);

    // 获取数据指针
    void *a_data = mat.storage()->data();

    // 使用类型分发器执行原地ReLU运算
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_unary_kernel<T, ReLUOp>(static_cast<const T *>(a_data), static_cast<T *>(a_data), mat.elements(),
                                       ReLUOp{}, 0);
    });

    CUDA_CHECK_ASYNC();
}

}  // namespace cuda
}  // namespace origin
