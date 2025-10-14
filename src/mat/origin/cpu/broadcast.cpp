#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> broadcast_to(const OriginMat &mat, const Shape &target_shape)
{
    // 简单的实现：创建目标形状的矩阵并复制数据
    auto result = std::make_unique<OriginMat>(target_shape, mat.dtype());

    // 复制数据到结果矩阵
    switch (mat.dtype())
    {
        case DataType::kFloat32:
        {
            const float *src_data = mat.data_ptr<float>();
            float *dst_data       = result->data_ptr<float>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % mat.elements()];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *src_data = mat.data_ptr<double>();
            double *dst_data       = result->data_ptr<double>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % mat.elements()];
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *src_data = mat.data_ptr<int32_t>();
            int32_t *dst_data       = result->data_ptr<int32_t>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % mat.elements()];
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *src_data = mat.data_ptr<int8_t>();
            int8_t *dst_data       = result->data_ptr<int8_t>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % mat.elements()];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type");
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
