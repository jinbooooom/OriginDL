#include <stdexcept>
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> matmul(const OriginMat &a, const OriginMat &b)
{
    // 检查数据类型匹配
    if (a.dtype() != b.dtype())
    {
        throw std::invalid_argument("Data type mismatch for matrix multiplication");
    }

    // 处理不同维度的矩阵乘法
    if (a.shape().size() == 2 && b.shape().size() == 2)
    {
        // 2D x 2D 矩阵乘法
        if (a.shape()[1] != b.shape()[0])
        {
            throw std::invalid_argument("Matrix dimensions must be compatible for multiplication");
        }
    }
    else if (a.shape().size() == 3 && b.shape().size() == 2)
    {
        // 3D x 2D 矩阵乘法：对3D张量的最后两个维度进行矩阵乘法
        if (a.shape()[2] != b.shape()[0])
        {
            throw std::invalid_argument("Matrix dimensions must be compatible for multiplication");
        }
    }
    else
    {
        throw std::invalid_argument("Matrix multiplication requires compatible dimensions");
    }

    // 计算结果形状
    Shape result_shape;
    if (a.shape().size() == 2)
    {
        result_shape = Shape({a.shape()[0], b.shape()[1]});
    }
    else if (a.shape().size() == 3)
    {
        result_shape = Shape({a.shape()[0], a.shape()[1], b.shape()[1]});
    }

    auto result = std::make_unique<OriginMat>(result_shape, a.dtype());

    switch (a.dtype())
    {
        case DataType::kFloat32:
        {
            const float *a_data = a.data_ptr<float>();
            const float *b_data = b.data_ptr<float>();
            float *c_data       = result->data_ptr<float>();

            if (a.shape().size() == 2)
            {
                // 2D x 2D 矩阵乘法
                for (size_t i = 0; i < a.shape()[0]; ++i)
                {
                    for (size_t j = 0; j < b.shape()[1]; ++j)
                    {
                        float sum = 0.0f;
                        for (size_t k = 0; k < a.shape()[1]; ++k)
                        {
                            sum += a_data[i * a.shape()[1] + k] * b_data[k * b.shape()[1] + j];
                        }
                        c_data[i * b.shape()[1] + j] = sum;
                    }
                }
            }
            else if (a.shape().size() == 3)
            {
                // 3D x 2D 矩阵乘法
                size_t batch_size = a.shape()[0];
                size_t m          = a.shape()[1];
                size_t k          = a.shape()[2];
                size_t n          = b.shape()[1];

                for (size_t batch = 0; batch < batch_size; ++batch)
                {
                    for (size_t i = 0; i < m; ++i)
                    {
                        for (size_t j = 0; j < n; ++j)
                        {
                            float sum = 0.0f;
                            for (size_t k_idx = 0; k_idx < k; ++k_idx)
                            {
                                size_t a_idx = batch * m * k + i * k + k_idx;
                                size_t b_idx = k_idx * n + j;
                                sum += a_data[a_idx] * b_data[b_idx];
                            }
                            size_t c_idx  = batch * m * n + i * n + j;
                            c_data[c_idx] = sum;
                        }
                    }
                }
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a_data = a.data_ptr<double>();
            const double *b_data = b.data_ptr<double>();
            double *c_data       = result->data_ptr<double>();

            if (a.shape().size() == 2)
            {
                // 2D x 2D 矩阵乘法
                for (size_t i = 0; i < a.shape()[0]; ++i)
                {
                    for (size_t j = 0; j < b.shape()[1]; ++j)
                    {
                        double sum = 0.0;
                        for (size_t k = 0; k < a.shape()[1]; ++k)
                        {
                            sum += a_data[i * a.shape()[1] + k] * b_data[k * b.shape()[1] + j];
                        }
                        c_data[i * b.shape()[1] + j] = sum;
                    }
                }
            }
            else if (a.shape().size() == 3)
            {
                // 3D x 2D 矩阵乘法
                size_t batch_size = a.shape()[0];
                size_t m          = a.shape()[1];
                size_t k          = a.shape()[2];
                size_t n          = b.shape()[1];

                for (size_t batch = 0; batch < batch_size; ++batch)
                {
                    for (size_t i = 0; i < m; ++i)
                    {
                        for (size_t j = 0; j < n; ++j)
                        {
                            double sum = 0.0;
                            for (size_t k_idx = 0; k_idx < k; ++k_idx)
                            {
                                size_t a_idx = batch * m * k + i * k + k_idx;
                                size_t b_idx = k_idx * n + j;
                                sum += a_data[a_idx] * b_data[b_idx];
                            }
                            size_t c_idx  = batch * m * n + i * n + j;
                            c_data[c_idx] = sum;
                        }
                    }
                }
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *a_data = a.data_ptr<int32_t>();
            const int32_t *b_data = b.data_ptr<int32_t>();
            int32_t *c_data       = result->data_ptr<int32_t>();

            if (a.shape().size() == 2)
            {
                // 2D x 2D 矩阵乘法
                for (size_t i = 0; i < a.shape()[0]; ++i)
                {
                    for (size_t j = 0; j < b.shape()[1]; ++j)
                    {
                        int32_t sum = 0;
                        for (size_t k = 0; k < a.shape()[1]; ++k)
                        {
                            sum += a_data[i * a.shape()[1] + k] * b_data[k * b.shape()[1] + j];
                        }
                        c_data[i * b.shape()[1] + j] = sum;
                    }
                }
            }
            else if (a.shape().size() == 3)
            {
                // 3D x 2D 矩阵乘法
                size_t batch_size = a.shape()[0];
                size_t m          = a.shape()[1];
                size_t k          = a.shape()[2];
                size_t n          = b.shape()[1];

                for (size_t batch = 0; batch < batch_size; ++batch)
                {
                    for (size_t i = 0; i < m; ++i)
                    {
                        for (size_t j = 0; j < n; ++j)
                        {
                            int32_t sum = 0;
                            for (size_t k_idx = 0; k_idx < k; ++k_idx)
                            {
                                size_t a_idx = batch * m * k + i * k + k_idx;
                                size_t b_idx = k_idx * n + j;
                                sum += a_data[a_idx] * b_data[b_idx];
                            }
                            size_t c_idx  = batch * m * n + i * n + j;
                            c_data[c_idx] = sum;
                        }
                    }
                }
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *a_data = a.data_ptr<int8_t>();
            const int8_t *b_data = b.data_ptr<int8_t>();
            int8_t *c_data       = result->data_ptr<int8_t>();

            if (a.shape().size() == 2)
            {
                // 2D x 2D 矩阵乘法
                for (size_t i = 0; i < a.shape()[0]; ++i)
                {
                    for (size_t j = 0; j < b.shape()[1]; ++j)
                    {
                        int32_t sum = 0;  // 使用int32_t避免溢出
                        for (size_t k = 0; k < a.shape()[1]; ++k)
                        {
                            sum += static_cast<int32_t>(a_data[i * a.shape()[1] + k]) *
                                   static_cast<int32_t>(b_data[k * b.shape()[1] + j]);
                        }
                        c_data[i * b.shape()[1] + j] = static_cast<int8_t>(sum);
                    }
                }
            }
            else if (a.shape().size() == 3)
            {
                // 3D x 2D 矩阵乘法
                size_t batch_size = a.shape()[0];
                size_t m          = a.shape()[1];
                size_t k          = a.shape()[2];
                size_t n          = b.shape()[1];

                for (size_t batch = 0; batch < batch_size; ++batch)
                {
                    for (size_t i = 0; i < m; ++i)
                    {
                        for (size_t j = 0; j < n; ++j)
                        {
                            int32_t sum = 0;  // 使用int32_t避免溢出
                            for (size_t k_idx = 0; k_idx < k; ++k_idx)
                            {
                                size_t a_idx = batch * m * k + i * k + k_idx;
                                size_t b_idx = k_idx * n + j;
                                sum += static_cast<int32_t>(a_data[a_idx]) * static_cast<int32_t>(b_data[b_idx]);
                            }
                            size_t c_idx  = batch * m * n + i * n + j;
                            c_data[c_idx] = static_cast<int8_t>(sum);
                        }
                    }
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for matrix multiplication");
    }

    return result;
}

}  // namespace cpu
}  // namespace origin
