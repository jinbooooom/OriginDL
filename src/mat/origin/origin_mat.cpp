/**
 * @file origin_mat.cpp
 * @brief OriginMat 类实现 - 封装层
 * 
 * ============================================================================
 * 文件功能说明
 * ============================================================================
 * 
 * 本文件是 OriginMat 类的实现，作为封装层负责设备分发和调用对应的 CPU/CUDA 实现。
 * 
 * 架构位置：
 * - origin_mat.cpp (本文件：封装层)
 *   ↓ 包含
 * - cuda_ops.cuh (所有 CUDA 算子的接口声明)
 *   ↓ 声明
 * - cuda_ops.cu (非计算类算子实现：clone、index_put)
 * - add.cu, divide.cu 等 (计算类算子实现)
 *   ↓ 都包含
 * - cuda_kernels.cuh (kernel 定义，只在 .cu 文件中使用)
 * 
 * 功能说明：
 * - 本文件只需包含 cuda_ops.cuh，即可使用所有 CUDA 算子
 * - 根据设备类型（CPU/CUDA）分发到对应的实现
 * - 不包含具体的 CUDA kernel 实现细节
 * 
 * ============================================================================
 */

#include "origin/mat/origin/origin_mat.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/cpu/factory.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/mat/scalar.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#    include "origin/mat/origin/cuda/cuda_ops.cuh"
#    include "origin/mat/origin/cuda/cuda_utils.cuh"
#    include "origin/mat/origin/cuda/factory.cuh"
#endif

// 前向声明
class OriginMat;

namespace origin
{

/**
 * @brief 统一的二元操作设备分发辅助函数（有返回值）
 * @param device_type 设备类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针
 * @param cpu_op CPU操作函数
 * @param cuda_op CUDA操作函数
 * @param op_name 操作名称（用于错误信息）
 * @return 操作结果
 */
template <typename OpFunc>
inline std::unique_ptr<Mat> device_dispatch_binary_op(DeviceType device_type,
                                                      const OriginMat &a,
                                                      const OriginMat &b,
                                                      OriginMat *out,
                                                      OpFunc cpu_op,
                                                      OpFunc cuda_op,
                                                      const char *op_name)
{
    if (device_type == DeviceType::kCPU)
    {
        return cpu_op(a, b, out);
    }
    else if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda_op(a, b, out);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for {}: {}", op_name, static_cast<int>(device_type));
    }
}

/**
 * @brief 统一的二元操作设备分发辅助函数（原地操作，无返回值）
 * @param device_type 设备类型
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针
 * @param cpu_op CPU操作函数
 * @param cuda_op CUDA操作函数
 * @param op_name 操作名称（用于错误信息）
 */
template <typename OpFunc>
inline void device_dispatch_binary_inplace_op(DeviceType device_type,
                                              const OriginMat &a,
                                              const OriginMat &b,
                                              OriginMat *out,
                                              OpFunc cpu_op,
                                              OpFunc cuda_op,
                                              const char *op_name)
{
    if (device_type == DeviceType::kCPU)
    {
        cpu_op(a, b, out);
    }
    else if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        cuda_op(a, b, out);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for {}: {}", op_name, static_cast<int>(device_type));
    }
}

/**
 * @brief 统一的一元操作设备分发辅助函数（有返回值）
 * @param device_type 设备类型
 * @param a 输入矩阵
 * @param out 输出矩阵指针
 * @param cpu_op CPU操作函数
 * @param cuda_op CUDA操作函数
 * @param op_name 操作名称（用于错误信息）
 * @return 操作结果
 */
template <typename OpFunc>
inline std::unique_ptr<Mat> device_dispatch_unary_op(DeviceType device_type,
                                                     const OriginMat &a,
                                                     OriginMat *out,
                                                     OpFunc cpu_op,
                                                     OpFunc cuda_op,
                                                     const char *op_name)
{
    if (device_type == DeviceType::kCPU)
    {
        return cpu_op(a, out);
    }
    else if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda_op(a, out);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for {}: {}", op_name, static_cast<int>(device_type));
    }
}

/**
 * @brief 统一的一元操作设备分发辅助函数（原地操作，无返回值）
 * @param device_type 设备类型
 * @param a 输入矩阵
 * @param out 输出矩阵指针
 * @param cpu_op CPU操作函数
 * @param cuda_op CUDA操作函数
 * @param op_name 操作名称（用于错误信息）
 */
template <typename OpFunc>
inline void device_dispatch_unary_inplace_op(DeviceType device_type,
                                             const OriginMat &a,
                                             OriginMat *out,
                                             OpFunc cpu_op,
                                             OpFunc cuda_op,
                                             const char *op_name)
{
    if (device_type == DeviceType::kCPU)
    {
        cpu_op(a, out);
    }
    else if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        cuda_op(a, out);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for {}: {}", op_name, static_cast<int>(device_type));
    }
}

/**
 * @brief 统一的标量操作设备分发辅助函数（有返回值）
 * @param device_type 设备类型
 * @param a 输入矩阵
 * @param scalar 标量参数
 * @param out 输出矩阵指针
 * @param cpu_op CPU操作函数
 * @param cuda_op CUDA操作函数
 * @param op_name 操作名称（用于错误信息）
 * @return 操作结果
 */
template <typename OpFunc>
inline std::unique_ptr<Mat> device_dispatch_scalar_op(DeviceType device_type,
                                                       const OriginMat &a,
                                                       const Scalar &scalar,
                                                       OriginMat *out,
                                                       OpFunc cpu_op,
                                                       OpFunc cuda_op,
                                                       const char *op_name)
{
    if (device_type == DeviceType::kCPU)
    {
        return cpu_op(a, scalar, out);
    }
    else if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda_op(a, scalar, out);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for {}: {}", op_name, static_cast<int>(device_type));
    }
}

// 构造函数实现
OriginMat::OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);
}

// 视图构造函数实现（用于创建视图，共享Storage）
OriginMat::OriginMat(std::shared_ptr<Storage> storage,
                     const Shape &shape,
                     const std::vector<size_t> &strides,
                     DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype), strides_(strides)
{
    utils::validate_shape(shape);
    // 验证strides大小与shape匹配
    if (unlikely(strides.size() != shape.size()))
    {
        THROW_INVALID_ARG("Strides size {} must match shape size {}", strides.size(), shape.size());
    }
}

// 为了向后兼容，保留一些构造函数
OriginMat::OriginMat(const Shape &shape, DataType dtype) : shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * element_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);
}

OriginMat::OriginMat(const Shape &shape, DataType dtype, Device device) : shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * element_size(dtype_);
    storage_    = Storage::create(size, device.type(), device.index());
}

// 两个核心工厂方法实现
std::unique_ptr<Mat> OriginMat::from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
{
    utils::validate_shape(shape);

    // 根据设备类型选择不同的实现方式
    if (options.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        // 对于CUDA设备，使用CUDA工厂方法
        return cuda::full(shape, scalar, options);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        // 对于CPU设备，使用CPU工厂方法
        return cpu::full(shape, scalar, options);
    }
}

std::unique_ptr<Mat> OriginMat::from_memory(const void *data,
                                            DataType user_dtype,
                                            const Shape &shape,
                                            const TensorOptions &options)
{
    utils::validate_shape(shape);

    // 根据设备类型选择不同的实现方式
    if (options.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        // 对于CUDA设备，使用CUDA工厂方法
        return cuda::from_memory(data, user_dtype, shape, options);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        // 对于CPU设备，使用CPU工厂方法
        return cpu::from_memory(data, user_dtype, shape, options);
    }
}

// Mat interface implementations - 委托给CPU模块
std::unique_ptr<Mat> OriginMat::clone() const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        // CPU 版本：直接调用 cpu_ops.cpp 中的 clone 实现
        // 该实现处理连续和非连续两种情况，支持按逻辑顺序拷贝非连续张量
        return cpu::clone(*this);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        // CUDA 版本：直接调用 cuda_ops.cu 中的 clone 实现
        // 该实现处理连续和非连续两种情况，支持按逻辑顺序拷贝非连续张量
        return cuda::clone(*this);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for clone: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::view(const Shape &new_shape) const
{
    // 验证元素总数必须匹配
    if (new_shape.elements() != shape_.elements())
    {
        THROW_INVALID_ARG("View: total elements must match. Original: {}, Target: {}", shape_.elements(),
                          new_shape.elements());
    }

    // 验证新形状是否有效
    utils::validate_shape(new_shape);

    // 创建视图：共享 Storage，只改变 shape 和 strides（零拷贝操作）
    // 使用现有的构造函数，它会自动计算新的 strides
    return std::make_unique<OriginMat>(storage_, new_shape, dtype_);
}

bool OriginMat::is_contiguous() const
{
    // 检查strides是否是标准的C风格连续（row-major）
    // 对于连续张量：strides[i] * shape[i] == strides[i-1] (对于i > 0)，且strides[ndim-1] == 1
    if (shape_.size() == 0)
    {
        return true;  // 标量张量总是连续的
    }

    // 计算标准的连续strides
    auto expected_strides = utils::compute_strides(shape_);

    // 比较实际strides和期望strides
    if (strides_.size() != expected_strides.size())
    {
        return false;
    }

    for (size_t i = 0; i < strides_.size(); ++i)
    {
        if (strides_[i] != expected_strides[i])
        {
            return false;
        }
    }

    return true;
}

std::unique_ptr<Mat> OriginMat::contiguous() const
{
    // 如果已经是连续的，返回视图（共享Storage）
    if (is_contiguous())
    {
        return view(shape_);
    }

    // 如果不是连续的，创建连续副本（深拷贝）
    return clone();
}

std::unique_ptr<Mat> OriginMat::reshape(const Shape &new_shape) const
{
    // 验证元素总数必须匹配
    if (new_shape.elements() != shape_.elements())
    {
        THROW_INVALID_ARG("Reshape: total elements must match. Original: {}, Target: {}", shape_.elements(),
                          new_shape.elements());
    }

    // 如果张量是连续的，使用view()创建视图（零拷贝）
    if (is_contiguous())
    {
        return view(new_shape);
    }

    // 如果张量不是连续的，需要创建连续副本后再reshape
    // 先创建连续副本，然后对连续副本使用view
    // 根本不会调用 cpu::reshape 和 cuda::reshape
    auto contiguous_mat = contiguous();
    return contiguous_mat->view(new_shape);
}

std::unique_ptr<Mat> OriginMat::transpose() const
{
    // 注意：当前实现使用数据转置（真正重新排列数据），而不是视图转置
    // 视图转置需要修改所有数据访问方法以支持strides，这是一个大工程
    // 未来可以优化为视图转置，但需要重构数据访问逻辑
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::transpose(*this);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::transpose(*this);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for transpose: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::permute(const std::vector<int> &dims) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::permute(*this, dims);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::permute(*this, dims);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for permute: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::operator+(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::add, cuda::add, "add");
}

void OriginMat::add_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::add, cuda::add,
                                      "add_inplace");
}

Mat &OriginMat::operator+=(const Mat &other)
{
    add_inplace(other);
    return *this;
}

std::unique_ptr<Mat> OriginMat::operator-(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::subtract, cuda::subtract,
                                     "subtract");
}

void OriginMat::sub_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::subtract, cuda::subtract,
                                      "sub_inplace");
}

Mat &OriginMat::operator-=(const Mat &other)
{
    sub_inplace(other);
    return *this;
}

// 比较运算符实现
std::unique_ptr<Mat> OriginMat::operator==(const Mat &threshold) const
{
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(threshold);
    return device_dispatch_binary_op(storage_->device_type(), *this, threshold_mat, nullptr, cpu::eq, cuda::eq, "eq");
}

std::unique_ptr<Mat> OriginMat::operator!=(const Mat &threshold) const
{
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(threshold);
    return device_dispatch_binary_op(storage_->device_type(), *this, threshold_mat, nullptr, cpu::ne, cuda::ne, "ne");
}

std::unique_ptr<Mat> OriginMat::operator<(const Mat &threshold) const
{
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(threshold);
    return device_dispatch_binary_op(storage_->device_type(), *this, threshold_mat, nullptr, cpu::lt, cuda::lt, "lt");
}

std::unique_ptr<Mat> OriginMat::operator<=(const Mat &threshold) const
{
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(threshold);
    return device_dispatch_binary_op(storage_->device_type(), *this, threshold_mat, nullptr, cpu::le, cuda::le, "le");
}

std::unique_ptr<Mat> OriginMat::operator>(const Mat &threshold) const
{
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(threshold);
    return device_dispatch_binary_op(storage_->device_type(), *this, threshold_mat, nullptr, cpu::gt, cuda::gt, "gt");
}

std::unique_ptr<Mat> OriginMat::operator>=(const Mat &threshold) const
{
    const OriginMat &threshold_mat = static_cast<const OriginMat &>(threshold);
    return device_dispatch_binary_op(storage_->device_type(), *this, threshold_mat, nullptr, cpu::ge, cuda::ge, "ge");
}

std::unique_ptr<Mat> OriginMat::operator*(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::multiply, cuda::multiply,
                                     "multiply");
}

void OriginMat::mul_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::multiply, cuda::multiply,
                                      "mul_inplace");
}

Mat &OriginMat::operator*=(const Mat &other)
{
    mul_inplace(other);
    return *this;
}

std::unique_ptr<Mat> OriginMat::operator/(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::divide, cuda::divide,
                                     "divide");
}

void OriginMat::div_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::divide, cuda::divide,
                                      "div_inplace");
}

Mat &OriginMat::operator/=(const Mat &other)
{
    div_inplace(other);
    return *this;
}

std::unique_ptr<Mat> OriginMat::operator-() const
{
    return device_dispatch_unary_op(storage_->device_type(), *this, nullptr, cpu::negate, cuda::negate, "negate");
}

void OriginMat::neg_inplace()
{
    device_dispatch_unary_inplace_op(storage_->device_type(), *this, this, cpu::negate, cuda::negate, "neg_inplace");
}

std::unique_ptr<Mat> OriginMat::square() const
{
    return device_dispatch_unary_op(storage_->device_type(), *this, nullptr, cpu::square, cuda::square, "square");
}

void OriginMat::square_inplace()
{
    device_dispatch_unary_inplace_op(storage_->device_type(), *this, this, cpu::square, cuda::square, "square_inplace");
}

void OriginMat::pow_inplace(const Scalar &exponent)
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        cpu::pow_inplace(*this, exponent);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        cuda::pow_inplace(*this, exponent);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for pow_inplace");
    }
}

std::unique_ptr<Mat> OriginMat::pow(const Scalar &exponent) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::pow(*this, exponent);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::pow(*this, exponent, nullptr);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for pow: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::matmul(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::matmul(*this, other_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::matmul(*this, other_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for matmul: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::sum(int axis, bool keepdim) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::sum(*this, axis, keepdim);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::sum(*this, axis, keepdim);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for sum: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::max(int axis) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::max(*this, axis);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::max(*this, axis);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for max: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::broadcast_to(const Shape &target_shape) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::broadcast_to(*this, target_shape);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::broadcast_to(*this, target_shape);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for broadcast_to: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::sum_to(const Shape &target_shape) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::sum_to(*this, target_shape);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::sum_to(*this, target_shape);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for sum_to: {}", static_cast<int>(storage_->device_type()));
    }
}

bool OriginMat::can_broadcast_to(const Shape &target_shape) const
{
    return utils::can_broadcast_to(shape_, target_shape);
}

// 形状和维度
Shape OriginMat::shape() const
{
    return shape_;
}

size_t OriginMat::elements() const
{
    return shape_.elements();
}

// 0维张量支持
bool OriginMat::is_scalar() const
{
    return shape_.is_scalar();
}

Scalar OriginMat::scalar_value() const
{
    if (!is_scalar())
    {
        THROW_INVALID_ARG("scalar_value() can only be called on scalar tensors (0-dimensional tensors)");
    }

    // 从存储中读取标量值
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return utils::compute::get_scalar_value(storage_->data(), dtype_);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        // 对于CUDA数据，需要先复制到CPU
        auto cpu_storage = storage_->to_device(DeviceType::kCPU, 0);
        return utils::compute::get_scalar_value(cpu_storage->data(), dtype_);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for scalar_value: {}", static_cast<int>(storage_->device_type()));
    }
}

// 数据访问
std::vector<float> OriginMat::to_vector() const
{
    // 如果数据在CUDA上，需要先复制到CPU
    if (storage_->device_type() == DeviceType::kCUDA)
    {
        auto cpu_storage = storage_->to_device(DeviceType::kCPU, 0);
        return utils::compute::convert_to_vector(cpu_storage->data(), shape_.elements(), dtype_);
    }
    else
    {
        return utils::compute::convert_to_vector(storage_->data(), shape_.elements(), dtype_);
    }
}

// 数学函数
std::unique_ptr<Mat> OriginMat::exp() const
{
    return device_dispatch_unary_op(storage_->device_type(), *this, nullptr, cpu::exp, cuda::exp, "exp");
}

void OriginMat::exp_inplace()
{
    device_dispatch_unary_inplace_op(storage_->device_type(), *this, this, cpu::exp, cuda::exp, "exp_inplace");
}

std::unique_ptr<Mat> OriginMat::relu() const
{
    return device_dispatch_unary_op(storage_->device_type(), *this, nullptr, cpu::relu, cuda::relu, "relu");
}

void OriginMat::relu_inplace()
{
    device_dispatch_unary_inplace_op(storage_->device_type(), *this, this, cpu::relu, cuda::relu, "relu_inplace");
}

std::unique_ptr<Mat> OriginMat::log() const
{
    return device_dispatch_unary_op(storage_->device_type(), *this, nullptr, cpu::log, cuda::log, "log");
}

void OriginMat::log_inplace()
{
    device_dispatch_unary_inplace_op(storage_->device_type(), *this, this, cpu::log, cuda::log, "log_inplace");
}

std::unique_ptr<Mat> OriginMat::sin() const
{
    // TODO: 实现sin函数
    THROW_RUNTIME_ERROR("sin function not implemented yet");
}

std::unique_ptr<Mat> OriginMat::cos() const
{
    // TODO: 实现cos函数
    THROW_RUNTIME_ERROR("cos function not implemented yet");
}

std::unique_ptr<Mat> OriginMat::sqrt() const
{
    return device_dispatch_unary_op(storage_->device_type(), *this, nullptr, cpu::sqrt, cuda::sqrt, "sqrt");
}

void OriginMat::sqrt_inplace()
{
    device_dispatch_unary_inplace_op(storage_->device_type(), *this, this, cpu::sqrt, cuda::sqrt, "sqrt_inplace");
}

// === 索引和选择操作 ===

Scalar OriginMat::index(std::initializer_list<size_t> indices) const
{
    if (unlikely(indices.size() != shape_.size()))
    {
        THROW_INVALID_ARG("Index count ({}) does not match tensor dimension ({}). Indices: {}, Shape: {}",
                          indices.size(), shape_.size(), "[indices]", shape_.to_string());
    }

    // 验证每个索引值并计算内存偏移（使用 strides，支持非连续内存）
    size_t offset = 0;
    size_t i      = 0;
    for (auto idx : indices)
    {
        if (unlikely(idx >= shape_[i]))
        {
            THROW_INVALID_ARG("Index {} out of range for dimension {} (size: {}). Indices: {}, Shape: {}", idx, i,
                              shape_[i], "[indices]", shape_.to_string());
        }
        offset += idx * strides_[i];
        ++i;
    }

    if (storage_->device_type() == DeviceType::kCPU)
    {
        void *data_ptr = storage_->data();
        return device_common::TypeDispatcher::dispatch(dtype_, [&]<typename T>() -> Scalar {
            const T *data = static_cast<const T *>(data_ptr);
            return Scalar(data[offset]);
        });
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        CUDA_CHECK(cudaDeviceSynchronize());
        void *data_ptr = storage_->data();
        return device_common::TypeDispatcher::dispatch(dtype_, [&]<typename T>() -> Scalar {
            T value;
            const T *data = static_cast<const T *>(data_ptr);
            CUDA_CHECK(cudaMemcpy(&value, &data[offset], sizeof(T), cudaMemcpyDeviceToHost));
            return Scalar(value);
        });
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for index: {}", static_cast<int>(storage_->device_type()));
    }
}

void OriginMat::index_put(std::initializer_list<size_t> indices, const Scalar &value)
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        cpu::index_put(*this, indices, value);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        cuda::index_put(*this, indices, value);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for index_put: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::gather(const Mat &indices) const
{
    // 类型检查和转换
    const OriginMat *indices_mat = dynamic_cast<const OriginMat *>(&indices);
    if (!indices_mat)
    {
        THROW_RUNTIME_ERROR("gather: indices must be OriginMat type, got backend_type={}", indices.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::gather(*this, *indices_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::gather(*this, *indices_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for gather: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::one_hot(const Mat &indices, int num_classes) const
{
    // 类型检查和转换
    const OriginMat *indices_mat = dynamic_cast<const OriginMat *>(&indices);
    if (!indices_mat)
    {
        THROW_RUNTIME_ERROR("one_hot: indices must be OriginMat type, got backend_type={}", indices.backend_type());
    }

    // 根据设备类型选择实现
    if (indices_mat->device().type() == DeviceType::kCPU)
    {
        return cpu::one_hot(*indices_mat, num_classes);
    }
    else if (indices_mat->device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::one_hot(*indices_mat, num_classes);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for one_hot: {}", static_cast<int>(indices_mat->device().type()));
    }
}

std::unique_ptr<Mat> OriginMat::yolo_detect_forward(const Mat &conv_weight,
                                                    const Mat *conv_bias,
                                                    const Mat &grid,
                                                    const Mat &anchor_grid,
                                                    float stride,
                                                    int32_t num_anchors,
                                                    int32_t num_classes) const
{
    // 类型检查和转换
    const OriginMat *conv_weight_mat = dynamic_cast<const OriginMat *>(&conv_weight);
    if (!conv_weight_mat)
    {
        THROW_RUNTIME_ERROR("yolo_detect_forward: conv_weight must be OriginMat type, got backend_type={}",
                            conv_weight.backend_type());
    }
    const OriginMat *conv_bias_mat = nullptr;
    if (conv_bias != nullptr)
    {
        conv_bias_mat = dynamic_cast<const OriginMat *>(conv_bias);
        if (!conv_bias_mat)
        {
            THROW_RUNTIME_ERROR("yolo_detect_forward: conv_bias must be OriginMat type, got backend_type={}",
                                conv_bias->backend_type());
        }
    }
    const OriginMat *grid_mat = dynamic_cast<const OriginMat *>(&grid);
    if (!grid_mat)
    {
        THROW_RUNTIME_ERROR("yolo_detect_forward: grid must be OriginMat type, got backend_type={}",
                            grid.backend_type());
    }
    const OriginMat *anchor_grid_mat = dynamic_cast<const OriginMat *>(&anchor_grid);
    if (!anchor_grid_mat)
    {
        THROW_RUNTIME_ERROR("yolo_detect_forward: anchor_grid must be OriginMat type, got backend_type={}",
                            anchor_grid.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::yolo_detect_forward(*this, *conv_weight_mat, conv_bias_mat, *grid_mat, *anchor_grid_mat, stride,
                                        num_anchors, num_classes);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::yolo_detect_forward(*this, *conv_weight_mat, conv_bias_mat, *grid_mat, *anchor_grid_mat, stride,
                                         num_anchors, num_classes);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for yolo_detect_forward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

// 类型和设备
DataType OriginMat::dtype() const
{
    return dtype_;
}

std::unique_ptr<Mat> OriginMat::to(DataType target_type) const
{
    // 根据设备类型选择实现
    if (device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::convert_datatype(*this, target_type);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        return cpu::convert_datatype(*this, target_type);
    }
}

Device OriginMat::device() const
{
    return Device(storage_->device_type(), storage_->device_index());
}

std::unique_ptr<Mat> OriginMat::to_device(Device device) const
{
    auto new_storage = storage_->to_device(device.type(), device.index());
    return std::make_unique<OriginMat>(new_storage, shape_, dtype_);
}

// 调试
void OriginMat::print(const std::string &desc) const
{
    auto data_vec          = to_vector();
    std::string device_str = device().to_string();
    utils::visualize::print_origin_mat(desc, data_vec, shape_.dims(), dtype_, device_str);
}

int OriginMat::backend_type() const
{
    return ORIGIN_BACKEND_TYPE;  // ORIGIN backend (0)
}

// 工厂方法
std::unique_ptr<Mat> OriginMat::randn(const Shape &shape, const TensorOptions &options)
{
    if (options.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        // 先在CPU上创建随机张量，然后移动到CUDA
        auto cpu_options = TensorOptions(options.dtype()).device(DeviceType::kCPU);
        auto cpu_tensor  = cpu::randn(shape, cpu_options);
        return cpu_tensor->to_device(options.device());
#else
        THROW_RUNTIME_ERROR("CUDA support not enabled, cannot create CUDA tensor");
#endif
    }
    else
    {
        return cpu::randn(shape, options);
    }
}

std::unique_ptr<Mat> OriginMat::zeros(const Shape &shape, const TensorOptions &options)
{
    if (options.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::zeros(shape, options);
#else
        THROW_RUNTIME_ERROR("CUDA support not enabled, cannot create CUDA tensor");
#endif
    }
    else
    {
        return cpu::zeros(shape, options);
    }
}

std::unique_ptr<Mat> OriginMat::ones(const Shape &shape, const TensorOptions &options)
{
    if (options.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::ones(shape, options);
#else
        THROW_RUNTIME_ERROR("CUDA support not enabled, cannot create CUDA tensor");
#endif
    }
    else
    {
        return cpu::ones(shape, options);
    }
}

std::unique_ptr<Mat> OriginMat::full(const Shape &shape, float value, const TensorOptions &options)
{
    if (options.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::full(shape, value, options);
#else
        THROW_RUNTIME_ERROR("CUDA support not enabled, cannot create CUDA tensor");
#endif
    }
    else
    {
        return cpu::full(shape, value, options);
    }
}

// 移除模板实例化，使用工厂方法替代

// === 卷积相关操作实现 ===

std::unique_ptr<Mat> OriginMat::im2col(std::pair<int, int> kernel_size,
                                       std::pair<int, int> stride,
                                       std::pair<int, int> pad,
                                       bool to_matrix) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::im2col(*this, kernel_size, stride, pad, to_matrix);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::im2col(*this, kernel_size, stride, pad, to_matrix);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for im2col: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::col2im(const Shape &input_shape,
                                       std::pair<int, int> kernel_size,
                                       std::pair<int, int> stride,
                                       std::pair<int, int> pad,
                                       bool to_matrix) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::col2im(*this, input_shape, kernel_size, stride, pad, to_matrix);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::col2im(*this, input_shape, kernel_size, stride, pad, to_matrix);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for col2im: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::conv2d(const Mat &W,
                                       const Mat *b,
                                       std::pair<int, int> stride,
                                       std::pair<int, int> pad) const
{
    // 类型检查和转换
    const OriginMat *W_mat = dynamic_cast<const OriginMat *>(&W);
    if (!W_mat)
    {
        THROW_RUNTIME_ERROR("conv2d: W must be OriginMat type, got backend_type={}", W.backend_type());
    }
    const OriginMat *b_mat = nullptr;
    if (b != nullptr)
    {
        b_mat = dynamic_cast<const OriginMat *>(b);
        if (!b_mat)
        {
            THROW_RUNTIME_ERROR("conv2d: b must be OriginMat type, got backend_type={}", b->backend_type());
        }
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::conv2d(*this, *W_mat, b_mat, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::conv2d(*this, *W_mat, b_mat, stride, pad);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for conv2d: {}", static_cast<int>(storage_->device_type()));
    }
}

std::vector<std::unique_ptr<Mat>> OriginMat::conv2d_backward(const Mat &gy,
                                                             const Mat &x,
                                                             const Mat &W,
                                                             const Mat *b,
                                                             std::pair<int, int> stride,
                                                             std::pair<int, int> pad) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("conv2d_backward: gy must be OriginMat type, got backend_type={}", gy.backend_type());
    }
    const OriginMat *x_mat = dynamic_cast<const OriginMat *>(&x);
    if (!x_mat)
    {
        THROW_RUNTIME_ERROR("conv2d_backward: x must be OriginMat type, got backend_type={}", x.backend_type());
    }
    const OriginMat *W_mat = dynamic_cast<const OriginMat *>(&W);
    if (!W_mat)
    {
        THROW_RUNTIME_ERROR("conv2d_backward: W must be OriginMat type, got backend_type={}", W.backend_type());
    }
    const OriginMat *b_mat = nullptr;
    if (b != nullptr)
    {
        b_mat = dynamic_cast<const OriginMat *>(b);
        if (!b_mat)
        {
            THROW_RUNTIME_ERROR("conv2d_backward: b must be OriginMat type, got backend_type={}", b->backend_type());
        }
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::conv2d_backward(*gy_mat, *x_mat, *W_mat, b_mat, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::conv2d_backward(*gy_mat, *x_mat, *W_mat, b_mat, stride, pad);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for conv2d_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::avg_pool2d(std::pair<int, int> kernel_size,
                                           std::pair<int, int> stride,
                                           std::pair<int, int> pad) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::avg_pool2d(*this, kernel_size, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::avg_pool2d(*this, kernel_size, stride, pad);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for avg_pool2d: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::avg_pool2d_backward(const Mat &gy,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride,
                                                    std::pair<int, int> pad) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("avg_pool2d_backward: gy must be OriginMat type, got backend_type={}", gy.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::avg_pool2d_backward(*gy_mat, *this, kernel_size, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::avg_pool2d_backward(*gy_mat, *this, kernel_size, stride, pad);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for avg_pool2d_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::adaptive_avg_pool2d(std::pair<int, int> output_size) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::adaptive_avg_pool2d(*this, output_size);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::adaptive_avg_pool2d(*this, output_size);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for adaptive_avg_pool2d: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::adaptive_avg_pool2d_backward(const Mat &gy, std::pair<int, int> output_size) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("adaptive_avg_pool2d_backward: gy must be OriginMat type, got backend_type={}",
                            gy.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::adaptive_avg_pool2d_backward(*gy_mat, *this, output_size);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::adaptive_avg_pool2d_backward(*gy_mat, *this, output_size);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for adaptive_avg_pool2d_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::max_pool2d(std::pair<int, int> kernel_size,
                                           std::pair<int, int> stride,
                                           std::pair<int, int> pad,
                                           std::vector<size_t> &indices) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::max_pool2d(*this, kernel_size, stride, pad, indices);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::max_pool2d(*this, kernel_size, stride, pad, indices);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for max_pool2d: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::max_pool2d_backward(const Mat &gy,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride,
                                                    std::pair<int, int> pad,
                                                    const std::vector<size_t> &indices) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("max_pool2d_backward: gy must be OriginMat type, got backend_type={}", gy.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::max_pool2d_backward(*gy_mat, *this, kernel_size, stride, pad, indices);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::max_pool2d_backward(*gy_mat, *this, kernel_size, stride, pad, indices);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for max_pool2d_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

// === 归一化相关操作实现 ===

OriginMat::BatchNormResult OriginMat::batch_norm_forward(const Mat &gamma,
                                                         const Mat &beta,
                                                         const Mat &running_mean,
                                                         const Mat &running_var,
                                                         bool training,
                                                         float eps,
                                                         int num_dims) const
{
    // 类型检查和转换
    const OriginMat *gamma_mat = dynamic_cast<const OriginMat *>(&gamma);
    if (!gamma_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_forward: gamma must be OriginMat type, got backend_type={}",
                            gamma.backend_type());
    }
    const OriginMat *beta_mat = dynamic_cast<const OriginMat *>(&beta);
    if (!beta_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_forward: beta must be OriginMat type, got backend_type={}",
                            beta.backend_type());
    }
    const OriginMat *running_mean_mat = dynamic_cast<const OriginMat *>(&running_mean);
    if (!running_mean_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_forward: running_mean must be OriginMat type, got backend_type={}",
                            running_mean.backend_type());
    }
    const OriginMat *running_var_mat = dynamic_cast<const OriginMat *>(&running_var);
    if (!running_var_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_forward: running_var must be OriginMat type, got backend_type={}",
                            running_var.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        auto result = cpu::batch_norm_forward(*this, *gamma_mat, *beta_mat, *running_mean_mat, *running_var_mat,
                                              training, eps, num_dims);
        OriginMat::BatchNormResult ret;
        ret.y      = std::move(result.y);
        ret.mean   = std::move(result.mean);
        ret.var    = std::move(result.var);
        ret.x_norm = std::move(result.x_norm);
        return ret;
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        auto result = cuda::batch_norm_forward(*this, *gamma_mat, *beta_mat, *running_mean_mat, *running_var_mat,
                                               training, eps, num_dims);
        OriginMat::BatchNormResult ret;
        ret.y      = std::move(result.y);
        ret.mean   = std::move(result.mean);
        ret.var    = std::move(result.var);
        ret.x_norm = std::move(result.x_norm);
        return ret;
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for batch_norm_forward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::batch_norm(const Mat &gamma,
                                           const Mat &beta,
                                           const Mat &running_mean,
                                           const Mat &running_var,
                                           bool training,
                                           float eps,
                                           float momentum,
                                           int num_dims) const
{
    // 类型检查和转换
    const OriginMat *gamma_mat = dynamic_cast<const OriginMat *>(&gamma);
    if (!gamma_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm: gamma must be OriginMat type, got backend_type={}", gamma.backend_type());
    }
    const OriginMat *beta_mat = dynamic_cast<const OriginMat *>(&beta);
    if (!beta_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm: beta must be OriginMat type, got backend_type={}", beta.backend_type());
    }
    const OriginMat *running_mean_mat = dynamic_cast<const OriginMat *>(&running_mean);
    if (!running_mean_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm: running_mean must be OriginMat type, got backend_type={}",
                            running_mean.backend_type());
    }
    const OriginMat *running_var_mat = dynamic_cast<const OriginMat *>(&running_var);
    if (!running_var_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm: running_var must be OriginMat type, got backend_type={}",
                            running_var.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::batch_norm(*this, *gamma_mat, *beta_mat, *running_mean_mat, *running_var_mat, training, eps,
                               momentum, num_dims);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::batch_norm(*this, *gamma_mat, *beta_mat, *running_mean_mat, *running_var_mat, training, eps,
                                momentum, num_dims);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for batch_norm: {}", static_cast<int>(storage_->device_type()));
    }
}

std::vector<std::unique_ptr<Mat>> OriginMat::batch_norm_backward(const Mat &gy,
                                                                 const Mat &gamma,
                                                                 const Mat &saved_mean,
                                                                 const Mat &saved_var,
                                                                 const Mat &saved_x_norm,
                                                                 float eps,
                                                                 int num_dims) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_backward: gy must be OriginMat type, got backend_type={}", gy.backend_type());
    }
    const OriginMat *gamma_mat = dynamic_cast<const OriginMat *>(&gamma);
    if (!gamma_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_backward: gamma must be OriginMat type, got backend_type={}",
                            gamma.backend_type());
    }
    const OriginMat *saved_mean_mat = dynamic_cast<const OriginMat *>(&saved_mean);
    if (!saved_mean_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_backward: saved_mean must be OriginMat type, got backend_type={}",
                            saved_mean.backend_type());
    }
    const OriginMat *saved_var_mat = dynamic_cast<const OriginMat *>(&saved_var);
    if (!saved_var_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_backward: saved_var must be OriginMat type, got backend_type={}",
                            saved_var.backend_type());
    }
    const OriginMat *saved_x_norm_mat = dynamic_cast<const OriginMat *>(&saved_x_norm);
    if (!saved_x_norm_mat)
    {
        THROW_RUNTIME_ERROR("batch_norm_backward: saved_x_norm must be OriginMat type, got backend_type={}",
                            saved_x_norm.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::batch_norm_backward(*gy_mat, *this, *gamma_mat, *saved_mean_mat, *saved_var_mat, *saved_x_norm_mat,
                                        eps, num_dims);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::batch_norm_backward(*gy_mat, *this, *gamma_mat, *saved_mean_mat, *saved_var_mat, *saved_x_norm_mat,
                                         eps, num_dims);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for batch_norm_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

// === Dropout 相关操作实现 ===

std::unique_ptr<Mat> OriginMat::dropout(float p, bool training, Mat *mask) const
{
    // mask 参数的类型检查和转换
    OriginMat *mask_mat = nullptr;
    if (mask != nullptr)
    {
        mask_mat = dynamic_cast<OriginMat *>(mask);
        if (!mask_mat)
        {
            THROW_RUNTIME_ERROR("dropout: mask must be OriginMat type, got backend_type={}", mask->backend_type());
        }
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::dropout(*this, p, training, mask_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::dropout(*this, p, training, mask_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for dropout: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::dropout_backward(const Mat &gy, const Mat &mask) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("dropout_backward: gy must be OriginMat type, got backend_type={}", gy.backend_type());
    }
    const OriginMat *mask_mat = dynamic_cast<const OriginMat *>(&mask);
    if (!mask_mat)
    {
        THROW_RUNTIME_ERROR("dropout_backward: mask must be OriginMat type, got backend_type={}", mask.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::dropout_backward(*gy_mat, *mask_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::dropout_backward(*gy_mat, *mask_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for dropout_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

// === Upsample 相关操作实现 ===

std::unique_ptr<Mat> OriginMat::upsample(const Shape &output_shape, int scale_h, int scale_w) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::upsample(*this, output_shape, scale_h, scale_w);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::upsample(*this, output_shape, scale_h, scale_w);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for upsample: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::upsample_backward(const Mat &gy, const Shape &x_shape, int scale_h, int scale_w) const
{
    // 类型检查和转换
    const OriginMat *gy_mat = dynamic_cast<const OriginMat *>(&gy);
    if (!gy_mat)
    {
        THROW_RUNTIME_ERROR("upsample_backward: gy must be OriginMat type, got backend_type={}", gy.backend_type());
    }

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::upsample_backward(*gy_mat, x_shape, scale_h, scale_w);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::upsample_backward(*gy_mat, x_shape, scale_h, scale_w);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for upsample_backward: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

// === Cat 和 Split 相关操作实现 ===

std::unique_ptr<Mat> OriginMat::cat(const std::vector<const Mat *> &others, int dim) const
{
    if (others.empty())
    {
        // 如果没有其他输入，直接返回当前矩阵的副本
        return clone();
    }

    // 检查所有输入的后端类型是否相同
    int backend_type = this->backend_type();
    for (const auto *other : others)
    {
        if (unlikely(other->backend_type() != backend_type))
        {
            THROW_RUNTIME_ERROR("cat: all inputs must have same backend type, got {} and {}", backend_type,
                                other->backend_type());
        }
    }

    // 构建所有输入的列表（包括当前对象）
    std::vector<const OriginMat *> origin_inputs;
    origin_inputs.reserve(others.size() + 1);
    origin_inputs.push_back(this);

    for (const auto *other : others)
    {
        const OriginMat *origin_mat = dynamic_cast<const OriginMat *>(other);
        if (unlikely(!origin_mat))
        {
            THROW_RUNTIME_ERROR("cat: failed to cast to OriginMat");
        }
        origin_inputs.push_back(origin_mat);
    }

    // 根据设备类型调用对应的实现
    DeviceType device_type = this->device().type();
    if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::cat(origin_inputs, dim);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        return cpu::cat(origin_inputs, dim);
    }
}

std::vector<std::unique_ptr<Mat>> OriginMat::split(const std::vector<size_t> &split_sizes, int dim) const
{
    // 根据设备类型调用对应的实现
    DeviceType device_type = this->device().type();
    if (device_type == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::split(*this, split_sizes, dim);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        return cpu::split(*this, split_sizes, dim);
    }
}

}  // namespace origin
