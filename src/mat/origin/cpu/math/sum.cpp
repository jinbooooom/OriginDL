#include <stdexcept>
#include "origin/mat/origin/device_common/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

// 前向声明
float sum_all(const OriginMat &mat);

std::unique_ptr<OriginMat> sum(const OriginMat &mat, int axis, bool keepdim)
{
    if (axis == -1)
    {
        // 对所有元素求和，返回标量
        float sum_value = sum_all(mat);
        Shape result_shape;
        if (keepdim)
        {
            // keepdim=true时保持所有维度为1
            result_shape = Shape(std::vector<size_t>(mat.shape().size(), 1));
        }
        else
        {
            result_shape = Shape({1});  // 标量结果
        }
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

    // 计算结果形状
    std::vector<size_t> result_dims;
    if (keepdim)
    {
        // keepdim=true时，在axis位置插入1
        for (size_t i = 0; i < mat.shape().size(); ++i)
        {
            if (i == static_cast<size_t>(axis))
            {
                result_dims.push_back(1);
            }
            else
            {
                result_dims.push_back(mat.shape()[i]);
            }
        }
    }
    else
    {
        // keepdim=false时，移除指定轴
        for (size_t i = 0; i < mat.shape().size(); ++i)
        {
            if (i != static_cast<size_t>(axis))
            {
                result_dims.push_back(mat.shape()[i]);
            }
        }
    }
    Shape result_shape(result_dims);

    auto result = std::make_unique<OriginMat>(result_shape, mat.dtype());

    // 使用类型分发器执行轴求和操作
    device_common::TypeDispatcher::dispatch_void(
        mat.dtype(), [&]<typename T>() { AxisSumCompute::axis_sum<T>(mat, *result, axis); });

    return result;
}

float sum_all(const OriginMat &mat)
{
    // 使用类型分发器执行全元素求和
    return device_common::TypeDispatcher::dispatch(mat.dtype(), [&]<typename T>() -> float {
        T sum_val = AxisSumCompute::sum_all<T>(mat);
        return static_cast<float>(sum_val);
    });
}

}  // namespace cpu
}  // namespace origin