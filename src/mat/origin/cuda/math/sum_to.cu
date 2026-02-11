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
 * @brief CUDA sum_to算子实现（参考CPU版本）
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return sum_to结果矩阵
 */
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape)
{
    VALIDATE_CUDA_DEVICE(mat);

    const auto &input_shape = mat.shape();

    if (input_shape == target_shape)
    {
        return std::make_unique<OriginMat>(mat);
    }

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
        // 比如形状 (3, 4, 5)，需对维度 0 和 1 求和：
        // 先 sum(1) -> (3, 5)，再 sum(0) -> (5,)，如果先 sum(0) -> (4, 5)，再 sum(1) -> (4, 1)，实际对原维度 2 求和
        std::sort(sum_dims.begin(), sum_dims.end(), std::greater<int>());
        for (int dim : sum_dims)
        {
            auto sum_result = current->sum(dim);
            current         = std::unique_ptr<OriginMat>(static_cast<OriginMat *>(sum_result.release()));
        }

        // 求和后形状可能仍与目标不同，需要 reshape
        // 原因：sum(dim) 会移除维度，而不是保留为1
        // 示例：输入 (3, 4, 5)，目标 (3, 1)
        //   - sum(2): (3, 4, 5) -> (3, 4)
        //   - sum(1): (3, 4) -> (3,)
        //   - reshape: (3,) -> (3, 1)
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
