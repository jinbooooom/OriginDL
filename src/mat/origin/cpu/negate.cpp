#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cpu/cpu_kernels.h"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU取负算子实现
 * @param mat 输入矩阵
 * @return 取负运算结果矩阵
 */
std::unique_ptr<Mat> negate(const OriginMat &mat)
{
    // 输入验证
    VALIDATE_CPU_DEVICE(mat);

    // 创建结果张量
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype(), mat.device());

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data = result->storage()->data();

    // 直接调用一元内核 - 与CUDA保持一致
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        cpu_unary_kernel<T, NegOp>(static_cast<const T *>(a_data), 
                                  static_cast<T *>(c_data), 
                                  mat.elements(), NegOp{});
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
