#include <memory>
#include <vector>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU data_transpose算子实现（数据转置）
 * @details 使用数据转置策略：真正重新排列内存中的数据
 * 
 * 数据转置的特点：
 * - 需要分配新内存并复制数据
 * - 数据在内存中的顺序被重新排列
 * - 转置后的张量是连续的
 * - 适用于需要真正转置数据的操作（如矩阵乘法）
 * 
 * 注意：当前实现只转置最后两个维度
 */
std::unique_ptr<OriginMat> data_transpose(const OriginMat &mat)
{
    if (mat.shape().size() == 0)
    {
        // 0维张量（标量）：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 1)
    {
        // 一维张量：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 2)
    {
        // 二维张量：转置最后两个维度（数据转置，重新排列数据）
        Shape new_shape({mat.shape()[1], mat.shape()[0]});
        auto result = std::make_unique<OriginMat>(new_shape, mat.dtype());

        // 使用类型分发器执行转置操作
        device_common::TypeDispatcher::dispatch_void(
            mat.dtype(), [&]<typename T>() { TransposeCompute::transpose_2d<T>(mat, *result); });

        return result;
    }
    else
    {
        // 高维张量：转置最后两个维度
        // 计算新的形状
        std::vector<size_t> new_dims = mat.shape().dims();
        std::swap(new_dims[new_dims.size() - 2], new_dims[new_dims.size() - 1]);
        Shape new_shape(new_dims);

        auto result = std::make_unique<OriginMat>(new_shape, mat.dtype());

        const void *src_data = mat.storage()->data();
        void *dst_data       = result->storage()->data();

        // 高维转置：转置最后两个维度
        const size_t last_dim        = mat.shape()[mat.shape().size() - 1];
        const size_t second_last_dim = mat.shape()[mat.shape().size() - 2];
        const size_t outer_elements  = mat.elements() / (last_dim * second_last_dim);

        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            const T *src = static_cast<const T *>(src_data);
            T *dst       = static_cast<T *>(dst_data);
            TransposeCompute::transpose_nd<T>(src, dst, last_dim, second_last_dim, outer_elements);
        });

        return result;
    }
}

/**
 * @brief CPU view_transpose算子实现（视图转置）
 * @details 使用视图转置策略：只改变 shape 和 strides，不重新排列内存中的数据
 * 
 * 视图转置的特点：
 * - 零拷贝操作，性能高
 * - 数据在内存中的顺序保持不变
 * - 通过改变 strides 来"模拟"转置效果
 * - 转置后的张量是非连续的，如果需要进行元素级操作，需要先调用 contiguous()
 * 
 * 注意：当前实现只转置最后两个维度
 */
std::unique_ptr<OriginMat> view_transpose(const OriginMat &mat)
{
    if (mat.shape().size() == 0)
    {
        // 0维张量（标量）：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 1)
    {
        // 一维张量：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else
    {
        // 二维或高维张量：转置最后两个维度（视图转置）
        // 计算新的形状
        std::vector<size_t> new_dims = mat.shape().dims();
        std::swap(new_dims[new_dims.size() - 2], new_dims[new_dims.size() - 1]);
        Shape new_shape(new_dims);

        // 计算新的 strides：交换最后两个维度的 strides
        std::vector<size_t> new_strides = mat.strides();
        std::swap(new_strides[new_strides.size() - 2], new_strides[new_strides.size() - 1]);

        // 使用视图构造函数创建转置后的张量（共享 storage，只改变 shape 和 strides）
        return std::make_unique<OriginMat>(mat.storage(), new_shape, new_strides, mat.dtype());
    }
}

/**
 * @brief CPU transpose算子实现（默认使用视图转置）
 * @details 调用 view_transpose 实现视图转置
 * 
 * 视图转置的特点：
 * - 零拷贝操作，性能高
 * - 数据在内存中的顺序保持不变
 * - 通过改变 strides 来"模拟"转置效果
 * - 转置后的张量是非连续的，如果需要进行元素级操作，需要先调用 contiguous()
 */
std::unique_ptr<OriginMat> transpose(const OriginMat &mat)
{
#if 0
    return view_transpose(mat);
#else
    return data_transpose(mat);
#endif
}

}  // namespace cpu
}  // namespace origin