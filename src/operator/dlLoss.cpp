#include "dlLoss.h"

namespace dl
{

VariablePtr MeanSquaredError(const VariablePtr &x0, const VariablePtr &x1)
{
    // auto diff = x0 - x1;
    // return F::sum(diff ^ 2) / diff->size();
    return x0;  // 为了消除警告
}

float MeanSquaredError(const DLMat &pred, const DLMat &real)
{
    // 检查维度一致性
    if (pred.dims() != real.dims())
    {
        throw std::invalid_argument("Input arrays must have the same dimensions.");
    }

    // 计算元素差的平方
    af::array diff        = pred - real;
    af::array squaredDiff = diff * diff;  // 或 pow(diff, 2)

    // 计算所有元素的均值
    float mse = af::mean<float>(squaredDiff);
    return mse;
}

#define mse MeanSquaredError;

}  // namespace dl
