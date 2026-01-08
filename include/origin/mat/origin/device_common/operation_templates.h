#ifndef __ORIGIN_DL_OPERATION_TEMPLATES_H__
#define __ORIGIN_DL_OPERATION_TEMPLATES_H__

#include <cmath>
#include <type_traits>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"

// 在纯CPU环境中，__host__ __device__ 修饰符会导致编译错误。需要使用条件编译来处理这个问题。
#ifdef __CUDACC__
#    define ORIGIN_HOST_DEVICE __host__ __device__
#else
#    define ORIGIN_HOST_DEVICE
#endif

namespace origin
{

/**
 * @brief 加法操作
 */
struct AddOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T a, T b) const
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            // 布尔类型不支持加法，使用逻辑OR作为替代
            return a || b;
        }
        else
        {
            return a + b;  // 对于数值类型使用加法
        }
    }
};

/**
 * @brief 除法操作
 */
struct DivideOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T a, T b) const
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            // 布尔类型不支持除法操作
            THROW_UNSUPPORTED("Division is not supported for boolean tensors");
        }
        else
        {
            return a / b;  // 对于数值类型使用除法
        }
    }
};

/**
 * @brief 平方操作
 */
struct SquareOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T value) const
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            return value && value;  // 对于布尔类型使用逻辑AND
        }
        else
        {
            return value * value;  // 对于数值类型使用乘法
        }
    }
};

/**
 * @brief 减法操作
 */
struct SubtractOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T a, T b) const
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            // 布尔类型不支持减法，使用逻辑异或 (XOR) 作为替代
            return a != b;
        }
        else
        {
            return a - b;  // 对于数值类型使用减法
        }
    }
};

/**
 * @brief 乘法操作
 */
struct MultiplyOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T a, T b) const
    {
        if constexpr (std::is_same_v<T, bool>)
        {
            // 布尔类型乘法等同于逻辑AND
            return a && b;
        }
        else
        {
            return a * b;  // 对于数值类型使用乘法
        }
    }
};

/**
 * @brief 指数操作
 */
struct ExpOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T value) const
    {
        return std::exp(value);
    }
};

/**
 * @brief 对数操作
 */
struct LogOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T value) const
    {
        return std::log(value);
    }
};

/**
 * @brief 平方根操作
 */
struct SqrtOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T value) const
    {
        return std::sqrt(value);
    }
};

/**
 * @brief 幂操作
 */
struct PowOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T base, T exponent) const
    {
        return std::pow(base, exponent);
    }
};

/**
 * @brief 取负操作
 */
struct NegOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T value) const
    {
        return -value;
    }
};

/**
 * @brief ReLU 操作
 */
struct ReLUOp
{
    template <typename T>
    ORIGIN_HOST_DEVICE T operator()(T value) const
    {
        return (value > T(0)) ? value : T(0);
    }
};

/**
 * @brief 轴求和操作
 * @details 提供模板化的轴求和实现，减少重复的类型处理代码
 */
class AxisSumCompute
{
public:
    /**
     * @brief 执行轴求和操作
     * @tparam T 数据类型
     * @param src 输入矩阵
     * @param dst 结果矩阵
     * @param axis 求和轴
     */
    template <typename T>
    static void axis_sum(const OriginMat &src, OriginMat &dst, int axis)
    {
        const T *src_data = src.data_ptr<T>();
        T *dst_data       = dst.data_ptr<T>();

        const Shape &src_shape = src.shape();
        const Shape &dst_shape = dst.shape();

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

            // 计算线性索引并求和
            T sum_val = T(0);
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

                sum_val += src_data[src_linear_idx];
            }

            dst_data[dst_idx] = sum_val;
        }
    }

    /**
     * @brief 执行全元素求和
     * @tparam T 数据类型
     * @param src 输入矩阵
     * @return 求和结果
     */
    template <typename T>
    static T sum_all(const OriginMat &src)
    {
        const T *data = src.data_ptr<T>();
        T sum         = T(0);
        for (size_t i = 0; i < src.elements(); ++i)
        {
            sum += data[i];
        }
        return sum;
    }
};

/**
 * @brief 转置计算操作
 * @details 提供模板化的转置实现，减少重复的类型处理代码
 */
class TransposeCompute
{
public:
    /**
     * @brief 执行二维矩阵转置
     * @tparam T 数据类型
     * @param src 输入矩阵
     * @param dst 结果矩阵
     */
    template <typename T>
    static void transpose_2d(const OriginMat &src, OriginMat &dst)
    {
        const T *src_data = src.data_ptr<T>();
        T *dst_data       = dst.data_ptr<T>();

        const size_t rows = src.shape()[0];
        const size_t cols = src.shape()[1];

        for (size_t i = 0; i < rows; ++i)
        {
            for (size_t j = 0; j < cols; ++j)
            {
                dst_data[j * rows + i] = src_data[i * cols + j];
            }
        }
    }
};

/**
 * @brief 广播到指定形状计算操作
 * @details 提供模板化的广播实现，减少重复的类型处理代码
 */
class BroadcastToCompute
{
public:
    /**
     * @brief 执行广播到指定形状操作（维度感知版本）
     * @tparam T 数据类型
     * @param src 输入矩阵
     * @param dst 结果矩阵
     */
    template <typename T>
    static void broadcast_to(const OriginMat &src, OriginMat &dst)
    {
        const T *src_data = src.data_ptr<T>();
        T *dst_data       = dst.data_ptr<T>();

        const Shape &src_shape = src.shape();
        const Shape &dst_shape = dst.shape();

        // 计算源和目标的 strides（用于多维索引）
        std::vector<size_t> src_strides(src_shape.size());
        std::vector<size_t> dst_strides(dst_shape.size());

        size_t src_stride = 1;
        size_t dst_stride = 1;
        for (int i = static_cast<int>(src_shape.size()) - 1; i >= 0; --i)
        {
            src_strides[i] = src_stride;
            src_stride *= src_shape[i];
        }
        for (int i = static_cast<int>(dst_shape.size()) - 1; i >= 0; --i)
        {
            dst_strides[i] = dst_stride;
            dst_stride *= dst_shape[i];
        }

        // 对齐维度：从右到左对齐
        int src_ndim = static_cast<int>(src_shape.size());
        int dst_ndim = static_cast<int>(dst_shape.size());

        // 对每个输出元素计算对应的源索引
        for (size_t dst_idx = 0; dst_idx < dst_shape.elements(); ++dst_idx)
        {
            // 将线性索引转换为多维索引
            std::vector<size_t> dst_indices(dst_ndim);
            size_t temp = dst_idx;
            for (int i = dst_ndim - 1; i >= 0; --i)
            {
                dst_indices[i] = temp % dst_shape[i];
                temp /= dst_shape[i];
            }

            // 计算对应的源索引
            // 从右到左对齐维度（NumPy/PyTorch 广播规则）
            size_t src_idx = 0;
            for (int i = 0; i < dst_ndim; ++i)
            {
                // 从右到左对齐：dst 的最后一个维度对应 src 的最后一个维度
                int dst_dim_idx = dst_ndim - 1 - i;
                int src_dim_idx = src_ndim - 1 - i;  // 从右到左对齐

                if (src_dim_idx >= 0 && src_dim_idx < src_ndim)
                {
                    // 如果源维度大小为1，则索引为0（广播）
                    size_t src_dim_size = src_shape[src_dim_idx];
                    size_t src_index    = (src_dim_size == 1) ? 0 : dst_indices[dst_dim_idx];
                    src_idx += src_index * src_strides[src_dim_idx];
                }
            }

            dst_data[dst_idx] = src_data[src_idx];
        }
    }
};

/**
 * @brief 归约计算操作
 * @details 提供模板化的归约操作实现，如最大值、最小值等
 */
class ReductionCompute
{
public:
    /**
     * @brief 计算所有元素的最大值
     * @tparam T 数据类型
     * @param mat 输入矩阵
     * @return 最大值
     */
    template <typename T>
    static T max_all(const OriginMat &mat)
    {
        const T *data = mat.data_ptr<T>();
        T max_val     = data[0];
        for (size_t i = 1; i < mat.elements(); ++i)
        {
            max_val = std::max(max_val, data[i]);
        }
        return max_val;
    }

    /**
     * @brief 计算所有元素的最小值
     * @tparam T 数据类型
     * @param mat 输入矩阵
     * @return 最小值
     */
    template <typename T>
    static T min_all(const OriginMat &mat)
    {
        const T *data = mat.data_ptr<T>();
        T min_val     = data[0];
        for (size_t i = 1; i < mat.elements(); ++i)
        {
            min_val = std::min(min_val, data[i]);
        }
        return min_val;
    }
};

/**
 * @brief 类型转换计算操作
 * @details 提供模板化的类型转换实现，减少重复的类型处理代码
 */
class TypeConversionCompute
{
public:
    /**
     * @brief 执行类型转换
     * @tparam SrcT 源数据类型
     * @tparam DstT 目标数据类型
     * @param src 输入矩阵
     * @param dst 结果矩阵
     */
    template <typename SrcT, typename DstT>
    static void convert(const OriginMat &src, OriginMat &dst)
    {
        const SrcT *src_data = src.data_ptr<SrcT>();
        DstT *dst_data       = dst.data_ptr<DstT>();

        const size_t elements = src.elements();
        for (size_t i = 0; i < elements; ++i)
        {
            dst_data[i] = static_cast<DstT>(src_data[i]);
        }
    }
};

/**
 * @brief 广播计算模板
 * @details 提供通用的广播计算逻辑，支持标量广播和形状匹配
 */
class BroadcastCompute
{
public:
    /**
     * @brief 执行二元广播操作，即 a与 b中如果有一个是标量，另一个是矩阵，则将标量广播到矩阵的形状，然后进行元素级运算
     * @tparam T 数据类型
     * @tparam Op 操作函数对象类型
     * @param a 输入矩阵A
     * @param b 输入矩阵B
     * @param result 结果矩阵
     * @param op 操作函数对象
     */
    template <typename T, typename Op>
    static void binary_broadcast(const OriginMat &a, const OriginMat &b, OriginMat &result, Op op)
    {
        const T *a_data = a.data_ptr<T>();
        const T *b_data = b.data_ptr<T>();
        T *c_data       = result.data_ptr<T>();

        const size_t a_elements = a.elements();
        const size_t b_elements = b.elements();

        if (a_elements == b_elements)
        {
            // 形状相同，直接操作
            for (size_t i = 0; i < a_elements; ++i)
            {
                c_data[i] = op(a_data[i], b_data[i]);
            }
        }
        // TODO:项目的元素为1的情况看是否可以去掉，经过优化后，到达这里的矩阵a,b会具有相同的dtype、shape、device。
        // 下面dOperator已经验证，其它的还需要再验证下
        // Add
        else if (a_elements == 1)
        {
            // a是标量，广播到b的形状
            const T scalar = a_data[0];
            for (size_t i = 0; i < b_elements; ++i)
            {
                c_data[i] = op(scalar, b_data[i]);
            }
        }
        else if (b_elements == 1)
        {
            // b是标量，广播到a的形状
            const T scalar = b_data[0];
            for (size_t i = 0; i < a_elements; ++i)
            {
                c_data[i] = op(a_data[i], scalar);
            }
        }
        else
        {
            THROW_INVALID_ARG("Incompatible shapes for broadcast operation: {} vs {}", a.shape().to_string(),
                              b.shape().to_string());
        }
    }

    /**
     * @brief 执行一元操作
     * @tparam T 数据类型
     * @tparam Op 操作函数对象类型
     * @param src 输入矩阵
     * @param dst 结果矩阵
     * @param op 操作函数对象
     */
    template <typename T, typename Op>
    static void unary(const OriginMat &src, OriginMat &dst, Op op)
    {
        const T *src_data     = src.data_ptr<T>();
        T *dst_data           = dst.data_ptr<T>();
        const size_t elements = src.elements();

        for (size_t i = 0; i < elements; ++i)
        {
            dst_data[i] = op(src_data[i]);
        }
    }
};

}  // namespace origin

#endif  // __ORIGIN_DL_OPERATION_TEMPLATES_H__
