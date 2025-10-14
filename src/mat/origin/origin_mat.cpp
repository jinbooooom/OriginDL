#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>

namespace origin
{

// 构造函数实现
OriginMat::OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);
}

OriginMat::OriginMat(const Shape &shape, DataType dtype) : shape_(shape), dtype_(dtype)
{
    utils::validate_shape(shape);
    strides_ = utils::compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * utils::get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);
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
    return cpu::reshape(*this, new_shape);
}

std::unique_ptr<Mat> OriginMat::transpose() const
{
    return cpu::transpose(*this);
}

std::unique_ptr<Mat> OriginMat::operator+(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return cpu::add(*this, other_mat);
}

std::unique_ptr<Mat> OriginMat::operator-(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return cpu::subtract(*this, other_mat);
}

std::unique_ptr<Mat> OriginMat::operator*(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return cpu::multiply(*this, other_mat);
}

std::unique_ptr<Mat> OriginMat::operator/(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return cpu::divide(*this, other_mat);
}

std::unique_ptr<Mat> OriginMat::operator+(data_t scalar) const
{
    return cpu::add_scalar(*this, scalar);
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
    return cpu::multiply_scalar(*this, scalar);
}

std::unique_ptr<Mat> OriginMat::operator-() const
{
    return cpu::negate(*this);
}

std::unique_ptr<Mat> OriginMat::square() const
{
    return cpu::square(*this);
}

std::unique_ptr<Mat> OriginMat::pow(data_t exponent) const
{
    return cpu::pow(*this, exponent);
}

std::unique_ptr<Mat> OriginMat::matmul(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    return cpu::matmul(*this, other_mat);
}

std::unique_ptr<Mat> OriginMat::sum(int axis) const
{
    return cpu::sum(*this, axis);
}

std::unique_ptr<Mat> OriginMat::broadcast_to(const Shape &target_shape) const
{
    return cpu::broadcast_to(*this, target_shape);
}

std::unique_ptr<Mat> OriginMat::sum_to(const Shape &target_shape) const
{
    return cpu::sum_to(*this, target_shape);
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
    return utils::compute::convert_to_vector(storage_->data(), shape_.elements(), dtype_);
}

// 数学函数
std::unique_ptr<Mat> OriginMat::exp() const
{
    return cpu::exp(*this);
}

std::unique_ptr<Mat> OriginMat::log() const
{
    return cpu::log(*this);
}

std::unique_ptr<Mat> OriginMat::sin() const
{
    // TODO: 实现sin函数
    throw std::runtime_error("sin function not implemented yet");
}

std::unique_ptr<Mat> OriginMat::cos() const
{
    // TODO: 实现cos函数
    throw std::runtime_error("cos function not implemented yet");
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
    // TODO: 实现设备转换
    throw std::runtime_error("device conversion not implemented yet");
}

// 调试
void OriginMat::print(const std::string &desc) const
{
    auto data_vec = to_vector();
    utils::visualize::print_origin_mat(desc, data_vec, shape_.dims(), dtype_, "cpu");
}

int OriginMat::backend_type() const
{
    return 2;  // ORIGIN backend
}

// 工厂方法
std::unique_ptr<Mat> OriginMat::randn(const Shape &shape, const TensorOptions &options)
{
    return cpu::randn(shape, options);
}

std::unique_ptr<Mat> OriginMat::zeros(const Shape &shape, const TensorOptions &options)
{
    return cpu::zeros(shape, options);
}

std::unique_ptr<Mat> OriginMat::ones(const Shape &shape, const TensorOptions &options)
{
    return cpu::ones(shape, options);
}

std::unique_ptr<Mat> OriginMat::full(const Shape &shape, data_t value, const TensorOptions &options)
{
    return cpu::full(shape, value, options);
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
