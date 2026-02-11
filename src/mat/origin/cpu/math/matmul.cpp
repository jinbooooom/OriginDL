#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

namespace
{

/**
 * @brief 矩阵乘法核心实现（2D x 2D）
 * @tparam T 数据类型
 * @param a_data 矩阵A的数据指针
 * @param b_data 矩阵B的数据指针
 * @param c_data 结果矩阵C的数据指针
 * @param m A的行数
 * @param n B的列数
 * @param k A的列数和B的行数
 */
template <typename T>
void matmul_2d_impl(const T *a_data, const T *b_data, T *c_data, size_t m, size_t n, size_t k)
{
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            T sum = T(0);
            for (size_t k_idx = 0; k_idx < k; ++k_idx)
            {
                sum += a_data[i * k + k_idx] * b_data[k_idx * n + j];
            }
            c_data[i * n + j] = sum;
        }
    }
}

/**
 * @brief 矩阵乘法核心实现（3D x 2D）
 * @tparam T 数据类型
 * @param a_data 矩阵A的数据指针
 * @param b_data 矩阵B的数据指针
 * @param c_data 结果矩阵C的数据指针
 * @param batch_size 批量大小
 * @param m A的行数
 * @param n B的列数
 * @param k A的列数和B的行数
 */
template <typename T>
void matmul_3d_impl(const T *a_data, const T *b_data, T *c_data, size_t batch_size, size_t m, size_t n, size_t k)
{
    for (size_t batch = 0; batch < batch_size; ++batch)
    {
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                T sum = T(0);
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

}  // namespace

std::unique_ptr<OriginMat> matmul(const OriginMat &a, const OriginMat &b)
{
    // 检查数据类型匹配 - 使用分支预测优化
    if (unlikely(a.dtype() != b.dtype()))
    {
        THROW_INVALID_ARG("Data type mismatch for matrix multiplication: expected {} but got {}",
                          dtype_to_string(a.dtype()), dtype_to_string(b.dtype()));
    }

    // 处理不同维度的矩阵乘法
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // 确保至少是2维（如果已经是2维以上，保持不变）
    // 注意：这里假设MatMul::forward已经处理了0维和1维的情况
    // 但如果直接调用底层函数，我们需要处理
    if (a_shape.size() < 2 || b_shape.size() < 2)
    {
        THROW_INVALID_ARG("Matrix multiplication requires at least 2D tensors. A shape: {}, B shape: {}",
                          a_shape.to_string(), b_shape.to_string());
    }

    if (a_shape.size() == 2 && b_shape.size() == 2)
    {
        // 2D x 2D 矩阵乘法
        if (unlikely(a_shape[1] != b_shape[0]))
        {
            THROW_INVALID_ARG(
                "Matrix dimensions must be compatible for multiplication. A shape: {}, B shape: {}, A[1]={} != B[0]={}",
                a_shape.to_string(), b_shape.to_string(), a_shape[1], b_shape[0]);
        }
    }
    else if (a_shape.size() == 3 && b_shape.size() == 2)
    {
        // 3D x 2D 矩阵乘法：对3D张量的最后两个维度进行矩阵乘法
        if (a_shape[2] != b_shape[0])
        {
            THROW_INVALID_ARG(
                "Matrix dimensions must be compatible for multiplication. A shape: {}, B shape: {}, A[2]={} != B[0]={}",
                a_shape.to_string(), b_shape.to_string(), a_shape[2], b_shape[0]);
        }
    }
    else
    {
        THROW_INVALID_ARG("Matrix multiplication requires compatible dimensions. A shape: {}, B shape: {}",
                          a_shape.to_string(), b_shape.to_string());
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

    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result->storage()->data();

    device_common::TypeDispatcher::dispatch_void(a.dtype(), [&]<typename T>() {
        const T *a_ptr = static_cast<const T *>(a_data);
        const T *b_ptr = static_cast<const T *>(b_data);
        T *c_ptr       = static_cast<T *>(c_data);

        if (a.shape().size() == 2)
        {
            matmul_2d_impl<T>(a_ptr, b_ptr, c_ptr, a.shape()[0], b.shape()[1], a.shape()[1]);
        }
        else if (a.shape().size() == 3)
        {
            matmul_3d_impl<T>(a_ptr, b_ptr, c_ptr, a.shape()[0], a.shape()[1], b.shape()[1], a.shape()[2]);
        }
    });

    return result;
}

}  // namespace cpu
}  // namespace origin
