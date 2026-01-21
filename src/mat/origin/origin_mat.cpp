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
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#    include "origin/mat/origin/cuda/cuda_kernels.cuh"
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
    // 深拷贝：创建新的 Storage 并复制数据（真正的独立副本）
    size_t data_size = shape_.elements() * element_size(dtype_);
    auto new_storage = Storage::create(data_size, storage_->device_type(), storage_->device_index());

    // 根据设备类型复制数据
    if (storage_->device_type() == DeviceType::kCPU)
    {
        std::memcpy(new_storage->data(), storage_->data(), data_size);
    }
    else
    {
#ifdef WITH_CUDA
        // 先同步，确保所有之前的异步kernel操作完成
        // 然后再复制数据，确保复制的是最新的数据
        cudaDeviceSynchronize();
        cudaError_t err = cudaMemcpy(new_storage->data(), storage_->data(), data_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
            THROW_RUNTIME_ERROR("CUDA memory copy failed in clone: {}", cudaGetErrorString(err));
        }
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }

    // 创建新的 OriginMat，使用新的 Storage
    return std::make_unique<OriginMat>(new_storage, shape_, dtype_);
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
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::add, cuda::add, "add_inplace");
}

std::unique_ptr<Mat> OriginMat::operator-(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::subtract, cuda::subtract, "subtract");
}

void OriginMat::sub_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::subtract, cuda::subtract, "sub_inplace");
}

std::unique_ptr<Mat> OriginMat::operator*(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::multiply, cuda::multiply, "multiply");
}

void OriginMat::mul_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::multiply, cuda::multiply, "mul_inplace");
}

std::unique_ptr<Mat> OriginMat::operator/(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return device_dispatch_binary_op(storage_->device_type(), *this, other_mat, nullptr, cpu::divide, cuda::divide, "divide");
}

void OriginMat::div_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    device_dispatch_binary_inplace_op(storage_->device_type(), *this, other_mat, this, cpu::divide, cuda::divide, "div_inplace");
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

std::unique_ptr<Mat> OriginMat::sum(int axis) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::sum(*this, axis);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::sum(*this, axis);
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
std::vector<data_t> OriginMat::to_vector() const
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
    size_t i = 0;
    for (auto idx : indices)
    {
        if (unlikely(idx >= shape_[i]))
        {
            THROW_INVALID_ARG("Index {} out of range for dimension {} (size: {}). Indices: {}, Shape: {}",
                              idx, i, shape_[i], "[indices]", shape_.to_string());
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

void OriginMat::index_put(std::initializer_list<size_t> indices, const Scalar& value)
{
    if (unlikely(indices.size() != shape_.size()))
    {
        THROW_INVALID_ARG("Index count ({}) does not match tensor dimension ({}). Indices: {}, Shape: {}",
                          indices.size(), shape_.size(), "[indices]", shape_.to_string());
    }

    // 验证每个索引值并计算内存偏移（使用 strides，支持非连续内存）
    size_t offset = 0;
    size_t i = 0;
    for (auto idx : indices)
    {
        if (unlikely(idx >= shape_[i]))
        {
            THROW_INVALID_ARG("Index {} out of range for dimension {} (size: {}). Indices: {}, Shape: {}",
                              idx, i, shape_[i], "[indices]", shape_.to_string());
        }
        offset += idx * strides_[i];
        ++i;
    }

    if (storage_->device_type() == DeviceType::kCPU)
    {
        void *data_ptr = storage_->data();
        device_common::TypeDispatcher::dispatch_void(dtype_, [&]<typename T>() {
            T *data = static_cast<T *>(data_ptr);
            data[offset] = value.to<T>();
        });
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        void *data_ptr = storage_->data();
        device_common::TypeDispatcher::dispatch_void(dtype_, [&]<typename T>() {
            T val = value.to<T>();
            T *data = static_cast<T *>(data_ptr);
            cuda::launch_index_put_kernel<T>(data, offset, val);
        });
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for index_put: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::gather(const OriginMat &indices) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::gather(*this, indices);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::gather(*this, indices);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for gather: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::one_hot(const OriginMat &indices, int num_classes)
{
    // 根据设备类型选择实现
    if (indices.device().type() == DeviceType::kCPU)
    {
        return cpu::one_hot(indices, num_classes);
    }
    else if (indices.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::one_hot(indices, num_classes);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for one_hot: {}", static_cast<int>(indices.device().type()));
    }
}

std::unique_ptr<Mat> OriginMat::yolo_detect_forward(const OriginMat &conv_weight,
                                                     const OriginMat *conv_bias,
                                                     const OriginMat &grid,
                                                     const OriginMat &anchor_grid,
                                                     float stride,
                                                     int32_t num_anchors,
                                                     int32_t num_classes) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::yolo_detect_forward(*this, conv_weight, conv_bias, grid, anchor_grid, stride, num_anchors, num_classes);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::yolo_detect_forward(*this, conv_weight, conv_bias, grid, anchor_grid, stride, num_anchors, num_classes);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for yolo_detect_forward: {}", static_cast<int>(storage_->device_type()));
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
    return 2;  // ORIGIN backend
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

std::unique_ptr<Mat> OriginMat::full(const Shape &shape, data_t value, const TensorOptions &options)
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

std::unique_ptr<Mat> OriginMat::conv2d(const OriginMat &W,
                                       const OriginMat *b,
                                       std::pair<int, int> stride,
                                       std::pair<int, int> pad) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::conv2d(*this, W, b, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::conv2d(*this, W, b, stride, pad);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for conv2d: {}", static_cast<int>(storage_->device_type()));
    }
}

std::vector<std::unique_ptr<Mat>> OriginMat::conv2d_backward(const OriginMat &gy,
                                                             const OriginMat &x,
                                                             const OriginMat &W,
                                                             const OriginMat *b,
                                                             std::pair<int, int> stride,
                                                             std::pair<int, int> pad) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::conv2d_backward(gy, x, W, b, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::conv2d_backward(gy, x, W, b, stride, pad);
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

std::unique_ptr<Mat> OriginMat::avg_pool2d_backward(const OriginMat &gy,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride,
                                                    std::pair<int, int> pad) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::avg_pool2d_backward(gy, *this, kernel_size, stride, pad);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::avg_pool2d_backward(gy, *this, kernel_size, stride, pad);
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

std::unique_ptr<Mat> OriginMat::adaptive_avg_pool2d_backward(const OriginMat &gy, std::pair<int, int> output_size) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::adaptive_avg_pool2d_backward(gy, *this, output_size);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::adaptive_avg_pool2d_backward(gy, *this, output_size);
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

std::unique_ptr<Mat> OriginMat::max_pool2d_backward(const OriginMat &gy,
                                                    std::pair<int, int> kernel_size,
                                                    std::pair<int, int> stride,
                                                    std::pair<int, int> pad,
                                                    const std::vector<size_t> &indices) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::max_pool2d_backward(gy, *this, kernel_size, stride, pad, indices);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::max_pool2d_backward(gy, *this, kernel_size, stride, pad, indices);
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

OriginMat::BatchNormResult OriginMat::batch_norm_forward(const OriginMat &gamma,
                                                         const OriginMat &beta,
                                                         const OriginMat &running_mean,
                                                         const OriginMat &running_var,
                                                         bool training,
                                                         float eps,
                                                         int num_dims) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        auto result = cpu::batch_norm_forward(*this, gamma, beta, running_mean, running_var, training, eps, num_dims);
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
        auto result = cuda::batch_norm_forward(*this, gamma, beta, running_mean, running_var, training, eps, num_dims);
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

std::unique_ptr<Mat> OriginMat::batch_norm(const OriginMat &gamma,
                                           const OriginMat &beta,
                                           const OriginMat &running_mean,
                                           const OriginMat &running_var,
                                           bool training,
                                           float eps,
                                           float momentum,
                                           int num_dims) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::batch_norm(*this, gamma, beta, running_mean, running_var, training, eps, momentum, num_dims);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::batch_norm(*this, gamma, beta, running_mean, running_var, training, eps, momentum, num_dims);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for batch_norm: {}", static_cast<int>(storage_->device_type()));
    }
}

std::vector<std::unique_ptr<Mat>> OriginMat::batch_norm_backward(const OriginMat &gy,
                                                                 const OriginMat &gamma,
                                                                 const OriginMat &saved_mean,
                                                                 const OriginMat &saved_var,
                                                                 const OriginMat &saved_x_norm,
                                                                 float eps,
                                                                 int num_dims) const
{
    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::batch_norm_backward(gy, *this, gamma, saved_mean, saved_var, saved_x_norm, eps, num_dims);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::batch_norm_backward(gy, *this, gamma, saved_mean, saved_var, saved_x_norm, eps, num_dims);
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

}  // namespace origin
