#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> add(const OriginMat &a, const OriginMat &b)
{
    // 检查数据类型是否匹配
    if (a.dtype() != b.dtype())
    {
        throw std::invalid_argument("Data type mismatch for addition");
    }

    // 处理形状广播
    Shape result_shape;
    if (a.shape() == b.shape())
    {
        // 形状相同，直接相加
        result_shape = a.shape();
    }
    else
    {
        // 尝试广播 - 简化实现，只支持标量广播
        if (a.elements() == 1)
        {
            result_shape = b.shape();
        }
        else if (b.elements() == 1)
        {
            result_shape = a.shape();
        }
        else
        {
            throw std::invalid_argument("Shape mismatch for addition - only scalar broadcasting supported");
        }
    }

    auto result = std::make_unique<OriginMat>(result_shape, a.dtype());

    // 执行广播加法
    switch (a.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = a.data_ptr<float>();
            const float *b_data = b.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();

            if (a.shape() == b.shape())
            {
                // 形状相同，直接相加
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + b_data[i];
                }
            }
            else if (a.elements() == 1)
            {
                // a是标量，广播到b的形状
                float scalar = a_data[0];
                for (size_t i = 0; i < b.elements(); ++i)
                {
                    c_data[i] = scalar + b_data[i];
                }
            }
            else if (b.elements() == 1)
            {
                // b是标量，广播到a的形状
                float scalar = b_data[0];
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + scalar;
                }
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = a.data_ptr<double>();
            const double *b_data = b.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();

            if (a.shape() == b.shape())
            {
                // 形状相同，直接相加
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + b_data[i];
                }
            }
            else if (a.elements() == 1)
            {
                // a是标量，广播到b的形状
                double scalar = a_data[0];
                for (size_t i = 0; i < b.elements(); ++i)
                {
                    c_data[i] = scalar + b_data[i];
                }
            }
            else if (b.elements() == 1)
            {
                // b是标量，广播到a的形状
                double scalar = b_data[0];
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + scalar;
                }
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = a.data_ptr<int32_t>();
            const int32_t *b_data = b.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();

            if (a.shape() == b.shape())
            {
                // 形状相同，直接相加
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + b_data[i];
                }
            }
            else if (a.elements() == 1)
            {
                // a是标量，广播到b的形状
                int32_t scalar = a_data[0];
                for (size_t i = 0; i < b.elements(); ++i)
                {
                    c_data[i] = scalar + b_data[i];
                }
            }
            else if (b.elements() == 1)
            {
                // b是标量，广播到a的形状
                int32_t scalar = b_data[0];
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + scalar;
                }
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = a.data_ptr<int8_t>();
            const int8_t *b_data = b.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();

            if (a.shape() == b.shape())
            {
                // 形状相同，直接相加
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + b_data[i];
                }
            }
            else if (a.elements() == 1)
            {
                // a是标量，广播到b的形状
                int8_t scalar = a_data[0];
                for (size_t i = 0; i < b.elements(); ++i)
                {
                    c_data[i] = scalar + b_data[i];
                }
            }
            else if (b.elements() == 1)
            {
                // b是标量，广播到a的形状
                int8_t scalar = b_data[0];
                for (size_t i = 0; i < a.elements(); ++i)
                {
                    c_data[i] = a_data[i] + scalar;
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for addition");
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
