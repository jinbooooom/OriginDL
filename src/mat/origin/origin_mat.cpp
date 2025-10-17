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
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#    include "origin/mat/origin/cuda/cuda_ops.cuh"
#    include "origin/mat/origin/cuda/factory.cuh"
#endif

namespace origin
{

// 构造函数实现
OriginMat::OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);
}

// TODO：与下面的函数相似，可以抽象出一个函数
OriginMat::OriginMat(const Shape &shape, DataType dtype) : shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);
}

OriginMat::OriginMat(const Shape &shape, DataType dtype, Device device) : shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, device.type(), device.index());
}

template <typename T>
OriginMat::OriginMat(const std::vector<T> &data, const Shape &shape)
    : shape_(shape), dtype_(utils::get_data_type_from_template<T>())
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 复制数据
    size_t data_size = data.size() * sizeof(T);
    memcpy(storage_->data(), data.data(), data_size);
}

template <typename T>
OriginMat::OriginMat(T value, const Shape &shape) : shape_(shape), dtype_(utils::get_data_type_from_template<T>())
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 填充数据
    T *data_ptr = static_cast<T *>(storage_->data());
    for (size_t i = 0; i < shape_.elements(); ++i)
    {
        data_ptr[i] = value;
    }
}

template <typename T>
OriginMat::OriginMat(const std::vector<T> &data, const Shape &shape, const TensorOptions &options)
    : shape_(shape), dtype_(options.dtype())
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 复制数据
    size_t data_size = data.size() * sizeof(T);
    memcpy(storage_->data(), data.data(), data_size);
}

template <typename T>
OriginMat::OriginMat(T value, const Shape &shape, const TensorOptions &options) : shape_(shape), dtype_(options.dtype())
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 填充数据
    T *data_ptr = static_cast<T *>(storage_->data());
    for (size_t i = 0; i < shape_.elements(); ++i)
    {
        data_ptr[i] = value;
    }
}

// Mat interface implementations - 委托给CPU模块
std::unique_ptr<Mat> OriginMat::clone() const
{
    return std::make_unique<OriginMat>(storage_, shape_, dtype_);
}

std::unique_ptr<Mat> OriginMat::reshape(const Shape &new_shape) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::reshape(*this, new_shape);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::reshape(*this, new_shape);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for reshape: {}", static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::transpose() const
{
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

std::unique_ptr<Mat> OriginMat::operator+(data_t scalar) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::add_scalar(*this, scalar);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::add_scalar(*this, scalar);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for scalar addition: {}",
                            static_cast<int>(storage_->device_type()));
    }
}

std::unique_ptr<Mat> OriginMat::operator-(data_t scalar) const
{
    return cpu::subtract_scalar(*this, scalar);
}

std::unique_ptr<Mat> OriginMat::operator*(data_t scalar) const
{
    return cpu::multiply_scalar(*this, scalar);
}

std::unique_ptr<Mat> OriginMat::operator/(data_t scalar) const
{
    return cpu::divide_scalar(*this, scalar);
}

std::unique_ptr<Mat> OriginMat::add_scalar(data_t scalar) const
{
    return cpu::add_scalar(*this, scalar);
}

std::unique_ptr<Mat> OriginMat::mul_scalar(data_t scalar) const
{
    if (storage_->device_type() == DeviceType::kCPU)
    {
        return cpu::multiply_scalar(*this, scalar);
    }
    else if (storage_->device_type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        return cuda::multiply_scalar(*this, scalar);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported device type for scalar multiplication: {}",
                            static_cast<int>(storage_->device_type()));
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
    return cpu::log(*this);
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

// 统计函数
data_t OriginMat::sum_all() const
{
    return cpu::sum_all(*this);
}

data_t OriginMat::max_all() const
{
    return cpu::max_all(*this);
}

data_t OriginMat::min_all() const
{
    return cpu::min_all(*this);
}

data_t OriginMat::mean_all() const
{
    return cpu::mean_all(*this);
}

// 类型和设备
DataType OriginMat::dtype() const
{
    return dtype_;
}

std::unique_ptr<Mat> OriginMat::to(DataType target_type) const
{
    return cpu::convert_datatype(*this, target_type);
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

// 模板实例化
template OriginMat::OriginMat(const std::vector<float> &, const Shape &);
template OriginMat::OriginMat(const std::vector<double> &, const Shape &);
template OriginMat::OriginMat(const std::vector<int32_t> &, const Shape &);
template OriginMat::OriginMat(const std::vector<int8_t> &, const Shape &);

template OriginMat::OriginMat(float, const Shape &);
template OriginMat::OriginMat(double, const Shape &);
template OriginMat::OriginMat(int32_t, const Shape &);
template OriginMat::OriginMat(int8_t, const Shape &);

template OriginMat::OriginMat(const std::vector<float> &, const Shape &, const TensorOptions &);
template OriginMat::OriginMat(const std::vector<double> &, const Shape &, const TensorOptions &);
template OriginMat::OriginMat(const std::vector<int32_t> &, const Shape &, const TensorOptions &);
template OriginMat::OriginMat(const std::vector<int8_t> &, const Shape &, const TensorOptions &);

template OriginMat::OriginMat(float, const Shape &, const TensorOptions &);
template OriginMat::OriginMat(double, const Shape &, const TensorOptions &);
template OriginMat::OriginMat(int32_t, const Shape &, const TensorOptions &);
template OriginMat::OriginMat(int8_t, const Shape &, const TensorOptions &);

}  // namespace origin
