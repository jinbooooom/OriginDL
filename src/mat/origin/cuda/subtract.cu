#include <cuda_runtime.h>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_broadcast.cuh"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include "origin/utils/branch_prediction.h"

namespace origin
{
namespace cuda
{
/**
 * @brief subtract算子实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 减法结果矩阵
 */
std::unique_ptr<Mat> subtract(const OriginMat &a, const OriginMat &b)
{
    // 验证输入
    VALIDATE_SAME_DTYPE(a, b);
    VALIDATE_SAME_CUDA_DEVICE(a, b);

    // 计算广播形状
    Shape result_shape = origin::utils::compute::compute_broadcast_shape(a, b);
    auto result        = std::make_unique<OriginMat>(result_shape, a.dtype(), a.device());

    // 获取数据指针
    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result->storage()->data();

    if (a.shape() == b.shape())
    {
        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            launch_elementwise_kernel<T, SubtractOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data), 
                                                    static_cast<T *>(c_data), a.elements(), SubtractOp{}, 0);
        });
    }
    else if (a.elements() == 1 || b.elements() == 1)
    {
        dim3 block = get_optimal_block_size(result->elements());
        dim3 grid  = get_optimal_grid_size(result->elements(), block);

        device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
            launch_simple_broadcast_kernel<T, SubtractOp>(static_cast<const T *>(a_data), static_cast<const T *>(b_data), 
                                                         static_cast<T *>(c_data), a.elements(), b.elements(), 
                                                         result->elements(), SubtractOp{}, 0);
        });
    }
    else
    {
        // 复杂广播：需要计算步长信息
        THROW_RUNTIME_ERROR("Complex broadcasting not yet implemented for CUDA subtract operation");
    }

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
