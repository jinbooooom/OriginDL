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
 * @brief sqrt算子实现
 * @param mat 输入矩阵
 * @return 平方根运算结果矩阵
 */
std::unique_ptr<Mat> sqrt(const OriginMat &mat)
{
    // 验证输入
    VALIDATE_CUDA_DEVICE(mat);
    VALIDATE_FLOAT_DTYPE(mat);

    // 创建结果张量
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 直接调用一元内核
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        launch_unary_kernel<T, SqrtOp>(static_cast<const T *>(a_data), static_cast<T *>(c_data), mat.elements(),
                                       SqrtOp{}, 0);
    });

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
