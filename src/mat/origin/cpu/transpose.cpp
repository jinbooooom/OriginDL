#include <stdexcept>
#include "origin/mat/origin/cpu/operation_templates.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

/*
视图转置 vs 数据转置的区别
1. 视图转置（View Transpose）
定义：只改变张量的形状和步长（stride），不重新排列内存中的数据
特点：
零拷贝操作，性能高
数据在内存中的顺序保持不变
通过改变索引计算方式来"模拟"转置效果
适用于大多数情况，特别是深度学习框架中的转置操作
2. 数据转置（Data Transpose）
定义：真正重新排列内存中的数据
特点：
需要分配新内存并复制数据
数据在内存中的顺序被重新排列
性能开销较大
主要用于矩阵乘法等需要真正转置数据的操作
*/
// 视图转置，不重新排列数据。只转置最后两个维度。未来还需要完善，完善视图转置的逻辑
std::unique_ptr<OriginMat> transpose(const OriginMat &mat)
{
    if (mat.shape().size() == 1)
    {
        // 一维张量：转置后保持不变
        return std::make_unique<OriginMat>(mat);
    }
    else if (mat.shape().size() == 2)
    {
        // 二维张量：转置最后两个维度（数据转置，重新排列数据）
        // 注意：为了与PyTorch行为一致，这里使用数据转置而不是视图转置
        Shape new_shape({mat.shape()[1], mat.shape()[0]});
        auto result = std::make_unique<OriginMat>(new_shape, mat.dtype());

        // 使用类型分发器执行转置操作
        device_common::TypeDispatcher::dispatch_void(mat.dtype(),
                                      [&]<typename T>() { TransposeCompute::transpose_2d<T>(mat, *result); });

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

        // 对于高维张量，需要更复杂的转置逻辑
        // 这里暂时使用简单的逐元素复制，保持原有逻辑
        switch (mat.dtype())
        {
            case DataType::kFloat32:
            {
                const float *src = mat.data_ptr<float>();
                float *dst       = result->data_ptr<float>();

                // 高维转置：转置最后两个维度
                const size_t last_dim        = mat.shape()[mat.shape().size() - 1];
                const size_t second_last_dim = mat.shape()[mat.shape().size() - 2];
                const size_t outer_elements  = mat.elements() / (last_dim * second_last_dim);

                for (size_t outer = 0; outer < outer_elements; ++outer)
                {
                    for (size_t i = 0; i < second_last_dim; ++i)
                    {
                        for (size_t j = 0; j < last_dim; ++j)
                        {
                            size_t src_idx = outer * (last_dim * second_last_dim) + i * last_dim + j;
                            size_t dst_idx = outer * (last_dim * second_last_dim) + j * second_last_dim + i;
                            dst[dst_idx]   = src[src_idx];
                        }
                    }
                }
                break;
            }
            case DataType::kFloat64:
            {
                const double *src = mat.data_ptr<double>();
                double *dst       = result->data_ptr<double>();

                const size_t last_dim        = mat.shape()[mat.shape().size() - 1];
                const size_t second_last_dim = mat.shape()[mat.shape().size() - 2];
                const size_t outer_elements  = mat.elements() / (last_dim * second_last_dim);

                for (size_t outer = 0; outer < outer_elements; ++outer)
                {
                    for (size_t i = 0; i < second_last_dim; ++i)
                    {
                        for (size_t j = 0; j < last_dim; ++j)
                        {
                            size_t src_idx = outer * (last_dim * second_last_dim) + i * last_dim + j;
                            size_t dst_idx = outer * (last_dim * second_last_dim) + j * second_last_dim + i;
                            dst[dst_idx]   = src[src_idx];
                        }
                    }
                }
                break;
            }
            case DataType::kInt32:
            {
                const int32_t *src = mat.data_ptr<int32_t>();
                int32_t *dst       = result->data_ptr<int32_t>();

                const size_t last_dim        = mat.shape()[mat.shape().size() - 1];
                const size_t second_last_dim = mat.shape()[mat.shape().size() - 2];
                const size_t outer_elements  = mat.elements() / (last_dim * second_last_dim);

                for (size_t outer = 0; outer < outer_elements; ++outer)
                {
                    for (size_t i = 0; i < second_last_dim; ++i)
                    {
                        for (size_t j = 0; j < last_dim; ++j)
                        {
                            size_t src_idx = outer * (last_dim * second_last_dim) + i * last_dim + j;
                            size_t dst_idx = outer * (last_dim * second_last_dim) + j * second_last_dim + i;
                            dst[dst_idx]   = src[src_idx];
                        }
                    }
                }
                break;
            }
            case DataType::kInt8:
            {
                const int8_t *src = mat.data_ptr<int8_t>();
                int8_t *dst       = result->data_ptr<int8_t>();

                const size_t last_dim        = mat.shape()[mat.shape().size() - 1];
                const size_t second_last_dim = mat.shape()[mat.shape().size() - 2];
                const size_t outer_elements  = mat.elements() / (last_dim * second_last_dim);

                for (size_t outer = 0; outer < outer_elements; ++outer)
                {
                    for (size_t i = 0; i < second_last_dim; ++i)
                    {
                        for (size_t j = 0; j < last_dim; ++j)
                        {
                            size_t src_idx = outer * (last_dim * second_last_dim) + i * last_dim + j;
                            size_t dst_idx = outer * (last_dim * second_last_dim) + j * second_last_dim + i;
                            dst[dst_idx]   = src[src_idx];
                        }
                    }
                }
                break;
            }
            default:
                THROW_INVALID_ARG("Unsupported data type {} for transpose operation", dtype_to_string(mat.dtype()));
        }

        return result;
    }
}

}  // namespace cpu
}  // namespace origin