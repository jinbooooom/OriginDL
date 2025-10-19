#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

// 前向声明
data_t sum_all(const OriginMat &mat);

std::unique_ptr<OriginMat> sum(const OriginMat &mat, int axis)
{
    if (axis == -1)
    {
        // 对所有元素求和，返回标量
        data_t sum_value   = sum_all(mat);
        Shape result_shape = {1};  // 标量结果
        // 创建标量张量
        Scalar scalar_val(sum_value);
        TensorOptions options(mat.dtype());
        auto mat_result = OriginMat::from_scalar(scalar_val, result_shape, options);
        return std::unique_ptr<OriginMat>(static_cast<OriginMat *>(mat_result.release()));
    }

    // 验证轴的有效性
    if (axis < 0 || axis >= static_cast<int>(mat.shape().size()))
    {
        THROW_INVALID_ARG("Invalid axis {} for sum operation. Tensor has {} dimensions", axis, mat.shape().size());
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

    auto result = std::make_unique<OriginMat>(result_shape, mat.dtype());

    // 使用类型分发器执行轴求和操作
    device_common::TypeDispatcher::dispatch_void(
        mat.dtype(), [&]<typename T>() { AxisSumCompute::axis_sum<T>(mat, *result, axis); });

    return result;
}

data_t sum_all(const OriginMat &mat)
{
    // 使用类型分发器执行全元素求和
    return device_common::TypeDispatcher::dispatch(mat.dtype(), [&]<typename T>() -> data_t {
        T sum_val = AxisSumCompute::sum_all<T>(mat);
        return static_cast<data_t>(sum_val);
    });
}

}  // namespace cpu
}  // namespace origin