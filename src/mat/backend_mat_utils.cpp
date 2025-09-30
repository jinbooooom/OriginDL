#include "origin/mat/backend_mat_utils.h"

namespace origin
{
namespace utils
{
af::array BroadcastTo(const af::array &src, const af::dim4 &targetShape)
{
    // 获取源数组和目标形状的维度
    af::dim4 src_dims = src.dims();

    // 如果源数组已经是目标形状，则直接返回
    if (src_dims == targetShape)
    {
        return src;
    }

    // 处理标量或单元素数组的特殊情况
    if (src.elements() == 1)
    {
        // 如果是标量，直接创建目标形状的常量数组
        float value;
        src.host(&value);
        return af::constant(value, targetShape);
    }

    // 对于一般情况，我们需要逐维度处理
    // 首先确保源维度与目标维度兼容
    for (int i = 0; i < 4; i++)
    {
        if (src_dims[i] != 1 && src_dims[i] != targetShape[i])
        {
            throw std::runtime_error("Incompatible dimensions for broadcasting");
        }
    }

    // 计算每个维度上需要重复的次数
    unsigned repeat_0 = (src_dims[0] == 1) ? targetShape[0] : 1;
    unsigned repeat_1 = (src_dims[1] == 1) ? targetShape[1] : 1;
    unsigned repeat_2 = (src_dims[2] == 1) ? targetShape[2] : 1;
    unsigned repeat_3 = (src_dims[3] == 1) ? targetShape[3] : 1;

    // 使用 tile 函数进行广播
    return af::tile(src, repeat_0, repeat_1, repeat_2, repeat_3);
}

af::array SumTo(const af::array &src, const af::dim4 &targetShape)
{
    // 获取源数组的维度
    af::dim4 src_dims = src.dims();

    // 如果源数组已经是目标形状，则直接返回
    if (src_dims == targetShape)
    {
        return src;
    }

    // 检查目标形状的维度是否合法
    for (int i = 0; i < 4; i++)
    {
        if (targetShape[i] > src_dims[i])
        {
            throw std::runtime_error("Target shape cannot have dimensions larger than source array");
        }
        if (targetShape[i] != src_dims[i] && targetShape[i] != 1)
        {
            throw std::runtime_error("Target dimensions must be 1 or equal to source dimensions");
        }
    }

    // 收集需要求和的维度
    std::vector<int> sum_dims;
    for (int i = 0; i < 4; i++)
    {
        if (targetShape[i] == 1 && src_dims[i] > 1)
        {
            sum_dims.push_back(i);
        }
    }

    // 如果没有需要求和的维度，检查是否需要调整形状
    if (sum_dims.empty())
    {
        // 计算源数组和目标形状的元素总数
        size_t src_elements    = src.elements();
        size_t target_elements = 1;
        for (int i = 0; i < 4; i++)
        {
            if (targetShape[i] > 0)  // 避免乘以0
            {
                target_elements *= targetShape[i];
            }
        }

        // 如果元素数量相同，只需调整形状
        if (src_elements == target_elements)
        {
            return af::moddims(src, targetShape);
        }
        else
        {
            // 元素数量不同但没有需要求和的维度，这是一个错误
            throw std::runtime_error(
                "Source and target shapes have different number of elements but no dimensions to sum");
        }
    }

    // 一次性对所有需要求和的维度进行求和
    af::array result = src;
    for (int dim : sum_dims)
    {
        result = af::sum(result, dim);
    }

    // 确保结果的形状正确
    af::dim4 result_dims = result.dims();

    // 如果结果形状与目标形状不一致，需要调整
    if (result_dims != targetShape)
    {
        result = af::moddims(result, targetShape);
    }

    return result;
}

}  // namespace utils
}  // namespace origin