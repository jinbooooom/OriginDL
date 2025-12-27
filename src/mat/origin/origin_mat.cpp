#include "origin/mat/origin/origin_mat.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/core/tensor_impl.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/cpu/factory.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#    include "origin/mat/origin/cuda/cuda_ops.cuh"
#    include "origin/mat/origin/cuda/factory.cuh"
#endif

// 前向声明
class OriginMat;

// 前向声明 add_inplace 函数（在 add.cpp 和 add.cu 中定义）
namespace origin
{
namespace cpu
{
void add_inplace(::origin::OriginMat &a, const ::origin::OriginMat &b);
}
#ifdef WITH_CUDA
namespace cuda
{
void add_inplace(::origin::OriginMat &a, const ::origin::OriginMat &b);
}
#endif
}


namespace origin
{

// 构造函数实现
OriginMat::OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);
}

// 视图构造函数实现（用于创建视图，共享Storage）
OriginMat::OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, const std::vector<size_t> &strides, DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype), strides_(strides)
{
    utils::validate_shape(shape);
    // 验证strides大小与shape匹配
    if (strides.size() != shape.size())
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

// TODO：
std::unique_ptr<Mat> OriginMat::operator+(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::add(*this, other_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::add(*this, other_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for addition: {}", static_cast<int>(storage_->device_type()));
    }
}

void OriginMat::add_inplace(const Mat &other)
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    // 根据设备类型选择实现
    if (storage_->device_type() == DeviceType::kCPU)
    {
        cpu::add_inplace(*this, other_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        cuda::add_inplace(*this, other_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for add_inplace: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::operator-(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::subtract(*this, other_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::subtract(*this, other_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for subtraction: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::operator*(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::multiply(*this, other_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::multiply(*this, other_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for multiplication: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::operator/(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::divide(*this, other_mat);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::divide(*this, other_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for division: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::operator-() const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::negate(*this);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::negate(*this);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for negate: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::square() const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::square(*this);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::square(*this);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for square: {}", static_cast<int>(storage_->device_type()));
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
        return cuda::pow(*this, exponent);
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
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::exp(*this);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::exp(*this);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for exp: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::log() const
{
    // 自然对数运算（以 e 为底）
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::log(*this);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::log(*this);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for log: {}", static_cast<int>(storage_->device_type()));
    }
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
    return cpu::sqrt(*this);
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

std::unique_ptr<Mat> OriginMat::im2col(std::pair<int, int> kernel_size, std::pair<int, int> stride,
                                       std::pair<int, int> pad, bool to_matrix) const
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

std::unique_ptr<Mat> OriginMat::col2im(const Shape &input_shape, std::pair<int, int> kernel_size,
                                       std::pair<int, int> stride, std::pair<int, int> pad, bool to_matrix) const
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

std::unique_ptr<Mat> OriginMat::conv2d(const OriginMat &W, const OriginMat *b, std::pair<int, int> stride,
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

std::vector<std::unique_ptr<Mat>> OriginMat::conv2d_backward(const OriginMat &gy, const OriginMat &x, const OriginMat &W,
                                                              const OriginMat *b, std::pair<int, int> stride,
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

}  // namespace origin
