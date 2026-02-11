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
 * @brief CUDA类型转换实现
 * @details 在GPU上直接进行类型转换，避免CPU-CUDA数据传输，提升性能
 *
 * ============================================================================
 * 实现说明
 * ============================================================================
 *
 * 1. 使用CUDA内核直接在GPU上进行类型转换
 * 2. 支持所有数据类型之间的转换
 * 3. 使用双重类型分发处理不同的源类型和目标类型组合
 * 4. 与CPU版本保持一致的转换行为
 *
 * ============================================================================
 * 性能优势
 * ============================================================================
 *
 * 相比之前的实现（先复制到CPU，转换后再复制回CUDA）：
 * - 避免了两次CPU-CUDA内存传输
 * - 直接在GPU上转换，减少延迟
 * - 对于大张量，性能提升显著
 *
 * ============================================================================
 */
std::unique_ptr<Mat> convert_datatype(const OriginMat &mat, DataType target_type)
{
    // 验证输入
    VALIDATE_CUDA_DEVICE(mat);

    // 如果类型相同，直接返回拷贝
    if (target_type == mat.dtype())
    {
        return std::make_unique<OriginMat>(mat);
    }

    // 创建结果张量（相同形状，不同类型，相同设备）
    auto result = std::make_unique<OriginMat>(mat.shape(), target_type, mat.device());

    // 获取数据指针
    const void *src_data = mat.storage()->data();
    void *dst_data       = result->storage()->data();

    // 等待所有CUDA操作完成,才能转换
    cudaDeviceSynchronize();

    // 使用双重类型分发执行类型转换
    // 因为有两层对 dtype 做 switch case，所以需要两次分发
    device_common::TypeDispatcher::dispatch_void(/*一次分发*/ mat.dtype(), [&]<typename SrcT>() {
        device_common::TypeDispatcher::dispatch_void(/*二次分发*/ target_type, [&]<typename DstT>() {
            launch_type_conversion_kernel<SrcT, DstT>(static_cast<const SrcT *>(src_data),
                                                      static_cast<DstT *>(dst_data), mat.elements(), 0);
        });
    });

    CUDA_CHECK_ASYNC();

    return result;
}

}  // namespace cuda
}  // namespace origin
