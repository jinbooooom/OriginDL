#include <algorithm>
#include <limits>
#include <stdexcept>
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/**
 * @brief CPU max：沿指定轴计算最大值
 * @param mat 输入矩阵
 * @param axis 计算轴，-1 表示所有元素
 * @return 最大值结果矩阵
 */
std::unique_ptr<Mat> max(const OriginMat &mat, int axis)
{
    if (axis == -1)
    {
        // 对所有元素求最大值，返回标量
        auto result_shape = Shape{1};  // 标量结果
        auto result       = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());

        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            const T *data = mat.data_ptr<T>();
            T max_val     = data[0];
            for (size_t i = 1; i < mat.elements(); ++i)
            {
                max_val = std::max(max_val, data[i]);
            }
            T *result_data = result->data_ptr<T>();
            result_data[0] = max_val;
        });

        return result;
    }

    // 验证轴的有效性
    if (axis < 0 || axis >= static_cast<int>(mat.shape().size()))
    {
        THROW_INVALID_ARG("Invalid axis {} for max operation. Tensor has {} dimensions", axis, mat.shape().size());
    }

    // 计算结果形状：移除指定轴
    std::vector<size_t> result_dims;
    for (size_t i = 0; i < mat.shape().size(); ++i)
    {
        if (i != static_cast<size_t>(axis))
        {
            result_dims.push_back(mat.shape()[i]);
        }
    }
    Shape result_shape(result_dims);

    auto result = std::make_unique<OriginMat>(result_shape, mat.dtype(), mat.device());

    // 使用类型分发器执行轴最大值操作
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        const T *src_data = mat.data_ptr<T>();
        T *dst_data       = result->data_ptr<T>();

        const Shape &src_shape = mat.shape();
        const Shape &dst_shape = result_shape;

        // 计算每个输出位置的索引
        std::vector<size_t> src_indices(src_shape.size(), 0);
        std::vector<size_t> dst_indices(dst_shape.size(), 0);

        for (size_t dst_idx = 0; dst_idx < dst_shape.elements(); ++dst_idx)
        {
            // 将一维索引转换为多维索引
            size_t temp = dst_idx;
            for (int i = dst_shape.size() - 1; i >= 0; --i)
            {
                dst_indices[i] = temp % dst_shape[i];
                temp /= dst_shape[i];
            }

            // 构建源索引
            for (size_t i = 0; i < src_shape.size(); ++i)
            {
                if (i == static_cast<size_t>(axis))
                {
                    src_indices[i] = 0;  // 轴维度从0开始
                }
                else
                {
                    // 找到对应的输出维度索引
                    size_t output_dim = (i < static_cast<size_t>(axis)) ? i : i - 1;
                    src_indices[i]    = dst_indices[output_dim];
                }
            }

            // 计算线性索引并求最大值
            T max_val = std::numeric_limits<T>::lowest();
            for (size_t axis_val = 0; axis_val < src_shape[axis]; ++axis_val)
            {
                src_indices[axis] = axis_val;

                // 计算源线性索引
                size_t src_linear_idx = 0;
                size_t stride         = 1;
                for (int i = src_shape.size() - 1; i >= 0; --i)
                {
                    src_linear_idx += src_indices[i] * stride;
                    stride *= src_shape[i];
                }

                max_val = std::max(max_val, src_data[src_linear_idx]);
            }

            dst_data[dst_idx] = max_val;
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
