#include "origin/mat/origin/origin_mat.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include "origin/mat/origin/origin_mat_utils.h"

namespace origin
{

// 构造函数实现
OriginMat::OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype)
    : storage_(storage), shape_(shape), dtype_(dtype)
{
    validate_shape(shape);
    strides_ = compute_strides(shape);
}

OriginMat::OriginMat(const Shape &shape, DataType dtype) : shape_(shape), dtype_(dtype)
{
    validate_shape(shape);
    strides_ = compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);
}

template <typename T>
OriginMat::OriginMat(const std::vector<T> &data, const Shape &shape)
    : shape_(shape), dtype_(get_data_type_from_template<T>())
{
    validate_shape(shape);
    strides_ = compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 复制数据
    size_t data_size = data.size() * sizeof(T);
    memcpy(storage_->data(), data.data(), data_size);
}

template <typename T>
OriginMat::OriginMat(T value, const Shape &shape) : shape_(shape), dtype_(get_data_type_from_template<T>())
{
    validate_shape(shape);
    strides_ = compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * get_dtype_size(dtype_);
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
    validate_shape(shape);
    strides_ = compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 复制数据
    size_t data_size = data.size() * sizeof(T);
    memcpy(storage_->data(), data.data(), data_size);
}

template <typename T>
OriginMat::OriginMat(T value, const Shape &shape, const TensorOptions &options) : shape_(shape), dtype_(options.dtype())
{
    validate_shape(shape);
    strides_ = compute_strides(shape);

    // 创建存储
    size_t size = shape_.elements() * get_dtype_size(dtype_);
    storage_    = Storage::create(size, DeviceType::kCPU);

    // 填充数据
    T *data_ptr = static_cast<T *>(storage_->data());
    for (size_t i = 0; i < shape_.elements(); ++i)
    {
        data_ptr[i] = value;
    }
}

// Helper methods
void OriginMat::validate_shape(const Shape &shape)
{
    if (shape.elements() == 0)
    {
        throw std::invalid_argument("Shape cannot have zero elements");
    }
}

std::vector<size_t> OriginMat::compute_strides(const Shape &shape)
{
    std::vector<size_t> strides(shape.ndims());
    size_t stride = 1;
    for (int i = shape.ndims() - 1; i >= 0; --i)
    {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

size_t OriginMat::get_dtype_size(DataType dtype) const
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return sizeof(float);
        case DataType::kFloat64:
            return sizeof(double);
        case DataType::kInt32:
            return sizeof(int32_t);
        case DataType::kInt8:
            return sizeof(int8_t);
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

template <typename T>
DataType OriginMat::get_data_type_from_template() const
{
    if (std::is_same_v<T, float>)
        return DataType::kFloat32;
    if (std::is_same_v<T, double>)
        return DataType::kFloat64;
    if (std::is_same_v<T, int32_t>)
        return DataType::kInt32;
    if (std::is_same_v<T, int8_t>)
        return DataType::kInt8;
    throw std::invalid_argument("Unsupported template type");
}

// Mat interface implementations
std::unique_ptr<Mat> OriginMat::clone() const
{
    auto new_storage = Storage::create(storage_->size(), storage_->device_type(), storage_->device_index());
    memcpy(new_storage->data(), storage_->data(), storage_->size());
    return std::make_unique<OriginMat>(new_storage, shape_, dtype_);
}

std::unique_ptr<Mat> OriginMat::reshape(const Shape &new_shape) const
{
    if (new_shape.elements() != shape_.elements())
    {
        throw std::invalid_argument("Reshape: total elements must match");
    }
    return std::make_unique<OriginMat>(storage_, new_shape, dtype_);
}

std::unique_ptr<Mat> OriginMat::transpose() const
{
    if (shape_.ndims() != 2)
    {
        throw std::invalid_argument("Transpose only supported for 2D matrices");
    }
    Shape new_shape({shape_[1], shape_[0]});
    return std::make_unique<OriginMat>(storage_, new_shape, dtype_);
}

std::unique_ptr<Mat> OriginMat::operator+(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    if (shape_ != other_mat.shape_)
    {
        throw std::invalid_argument("Shape mismatch for addition");
    }
    if (dtype_ != other_mat.dtype_)
    {
        throw std::invalid_argument("Data type mismatch for addition");
    }

    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            const float *b = other_mat.data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] + b[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            const double *b = other_mat.data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] + b[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for addition");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator-(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    if (shape_ != other_mat.shape_)
    {
        throw std::invalid_argument("Shape mismatch for subtraction");
    }
    if (dtype_ != other_mat.dtype_)
    {
        throw std::invalid_argument("Data type mismatch for subtraction");
    }

    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            const float *b = other_mat.data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] - b[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            const double *b = other_mat.data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] - b[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for subtraction");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator*(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    if (shape_ != other_mat.shape_)
    {
        throw std::invalid_argument("Shape mismatch for multiplication");
    }
    if (dtype_ != other_mat.dtype_)
    {
        throw std::invalid_argument("Data type mismatch for multiplication");
    }

    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            const float *b = other_mat.data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] * b[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            const double *b = other_mat.data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] * b[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for multiplication");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator/(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);
    if (shape_ != other_mat.shape_)
    {
        throw std::invalid_argument("Shape mismatch for division");
    }
    if (dtype_ != other_mat.dtype_)
    {
        throw std::invalid_argument("Data type mismatch for division");
    }

    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            const float *b = other_mat.data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] / b[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            const double *b = other_mat.data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] / b[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for division");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator+(data_t scalar) const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            float s        = static_cast<float>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] + s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            double s        = static_cast<double>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] + s;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for scalar addition");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator-(data_t scalar) const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            float s        = static_cast<float>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] - s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            double s        = static_cast<double>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] - s;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for scalar subtraction");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator*(data_t scalar) const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            float s        = static_cast<float>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] * s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            double s        = static_cast<double>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] * s;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for scalar multiplication");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::operator/(data_t scalar) const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            float s        = static_cast<float>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] / s;
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            double s        = static_cast<double>(scalar);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] / s;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for scalar division");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::add_scalar(data_t scalar) const
{
    return operator+(scalar);
}

std::unique_ptr<Mat> OriginMat::mul_scalar(data_t scalar) const
{
    return operator*(scalar);
}

std::unique_ptr<Mat> OriginMat::operator-() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = -a[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = -a[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for negation");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::square() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] * a[i];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = a[i] * a[i];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for square");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::pow(data_t exponent) const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            float exp      = static_cast<float>(exponent);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::pow(a[i], exp);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            double exp      = static_cast<double>(exponent);
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::pow(a[i], exp);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for power");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::matmul(const Mat &other) const
{
    const OriginMat &other_mat = static_cast<const OriginMat &>(other);

    if (shape_.ndims() != 2 || other_mat.shape_.ndims() != 2)
    {
        throw std::invalid_argument("Matrix multiplication requires 2D matrices");
    }
    if (shape_[1] != other_mat.shape_[0])
    {
        throw std::invalid_argument("Matrix dimensions must be compatible for multiplication");
    }
    if (dtype_ != other_mat.dtype_)
    {
        throw std::invalid_argument("Data type mismatch for matrix multiplication");
    }

    Shape result_shape({shape_[0], other_mat.shape_[1]});
    auto result = std::make_unique<OriginMat>(result_shape, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            const float *b = other_mat.data_ptr<float>();
            float *c       = result->data_ptr<float>();

            for (size_t i = 0; i < shape_[0]; ++i)
            {
                for (size_t j = 0; j < other_mat.shape_[1]; ++j)
                {
                    float sum = 0.0f;
                    for (size_t k = 0; k < shape_[1]; ++k)
                    {
                        sum += a[i * shape_[1] + k] * b[k * other_mat.shape_[1] + j];
                    }
                    c[i * other_mat.shape_[1] + j] = sum;
                }
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            const double *b = other_mat.data_ptr<double>();
            double *c       = result->data_ptr<double>();

            for (size_t i = 0; i < shape_[0]; ++i)
            {
                for (size_t j = 0; j < other_mat.shape_[1]; ++j)
                {
                    double sum = 0.0;
                    for (size_t k = 0; k < shape_[1]; ++k)
                    {
                        sum += a[i * shape_[1] + k] * b[k * other_mat.shape_[1] + j];
                    }
                    c[i * other_mat.shape_[1] + j] = sum;
                }
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for matrix multiplication");
    }

    return result;
}

// 实现缺少的方法
std::unique_ptr<Mat> OriginMat::sum(int axis) const
{
    // 简单的实现：返回所有元素的和
    data_t sum_value   = sum_all();
    Shape result_shape = {1};  // 标量结果
    return std::make_unique<OriginMat>(sum_value, result_shape);
}

std::unique_ptr<Mat> OriginMat::broadcast_to(const Shape &target_shape) const
{
    // 简单的实现：创建目标形状的矩阵并复制数据
    auto result = std::make_unique<OriginMat>(target_shape, dtype_);

    // 复制数据到结果矩阵
    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *src_data = data_ptr<float>();
            float *dst_data       = result->data_ptr<float>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % shape_.elements()];
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *src_data = data_ptr<double>();
            double *dst_data       = result->data_ptr<double>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % shape_.elements()];
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *src_data = data_ptr<int32_t>();
            int32_t *dst_data       = result->data_ptr<int32_t>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % shape_.elements()];
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *src_data = data_ptr<int8_t>();
            int8_t *dst_data       = result->data_ptr<int8_t>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                dst_data[i] = src_data[i % shape_.elements()];
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::sum_to(const Shape &target_shape) const
{
    // 简单的实现：将当前矩阵的所有元素求和，然后广播到目标形状
    data_t sum_value = sum_all();

    // 创建结果矩阵
    auto result = std::make_unique<OriginMat>(target_shape, dtype_);

    // 填充结果矩阵
    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            float *result_data = result->data_ptr<float>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                result_data[i] = static_cast<float>(sum_value);
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *result_data = result->data_ptr<double>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                result_data[i] = static_cast<double>(sum_value);
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *result_data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                result_data[i] = static_cast<int32_t>(sum_value);
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *result_data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < target_shape.elements(); ++i)
            {
                result_data[i] = static_cast<int8_t>(sum_value);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type");
    }

    return result;
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
    std::vector<data_t> result(shape_.elements());

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *data = data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *data = data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        case DataType::kInt32:
        {
            const int32_t *data = data_ptr<int32_t>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        case DataType::kInt8:
        {
            const int8_t *data = data_ptr<int8_t>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                result[i] = static_cast<data_t>(data[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for vector conversion");
    }

    return result;
}

// 数学函数
std::unique_ptr<Mat> OriginMat::exp() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::exp(a[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::exp(a[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for exp");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::log() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::log(a[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::log(a[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for log");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::sin() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::sin(a[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::sin(a[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for sin");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::cos() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::cos(a[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::cos(a[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for cos");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::sqrt() const
{
    auto result = std::make_unique<OriginMat>(shape_, dtype_);

    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *a = data_ptr<float>();
            float *c       = result->data_ptr<float>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::sqrt(a[i]);
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *a = data_ptr<double>();
            double *c       = result->data_ptr<double>();
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                c[i] = std::sqrt(a[i]);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for sqrt");
    }

    return result;
}

// 统计函数
data_t OriginMat::sum_all() const
{
    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *data = data_ptr<float>();
            float sum         = 0.0f;
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        case DataType::kFloat64:
        {
            const double *data = data_ptr<double>();
            double sum         = 0.0;
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        case DataType::kInt32:
        {
            const int32_t *data = data_ptr<int32_t>();
            int32_t sum         = 0;
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        case DataType::kInt8:
        {
            const int8_t *data = data_ptr<int8_t>();
            int8_t sum         = 0;
            for (size_t i = 0; i < shape_.elements(); ++i)
            {
                sum += data[i];
            }
            return static_cast<data_t>(sum);
        }
        default:
            throw std::invalid_argument("Unsupported data type for sum_all");
    }
}

data_t OriginMat::max_all() const
{
    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *data = data_ptr<float>();
            float max_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        case DataType::kFloat64:
        {
            const double *data = data_ptr<double>();
            double max_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        case DataType::kInt32:
        {
            const int32_t *data = data_ptr<int32_t>();
            int32_t max_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        case DataType::kInt8:
        {
            const int8_t *data = data_ptr<int8_t>();
            int8_t max_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                max_val = std::max(max_val, data[i]);
            }
            return static_cast<data_t>(max_val);
        }
        default:
            throw std::invalid_argument("Unsupported data type for max_all");
    }
}

data_t OriginMat::min_all() const
{
    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *data = data_ptr<float>();
            float min_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        case DataType::kFloat64:
        {
            const double *data = data_ptr<double>();
            double min_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        case DataType::kInt32:
        {
            const int32_t *data = data_ptr<int32_t>();
            int32_t min_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        case DataType::kInt8:
        {
            const int8_t *data = data_ptr<int8_t>();
            int8_t min_val     = data[0];
            for (size_t i = 1; i < shape_.elements(); ++i)
            {
                min_val = std::min(min_val, data[i]);
            }
            return static_cast<data_t>(min_val);
        }
        default:
            throw std::invalid_argument("Unsupported data type for min_all");
    }
}

data_t OriginMat::mean_all() const
{
    return sum_all() / static_cast<data_t>(shape_.elements());
}

// 类型和设备
DataType OriginMat::dtype() const
{
    return dtype_;
}

std::unique_ptr<Mat> OriginMat::to(DataType target_type) const
{
    if (target_type == dtype_)
    {
        return clone();
    }

    auto result = std::make_unique<OriginMat>(shape_, target_type);

    // 类型转换
    switch (dtype_)
    {
        case DataType::kFloat32:
        {
            const float *src = data_ptr<float>();
            switch (target_type)
            {
                case DataType::kFloat64:
                {
                    double *dst = result->data_ptr<double>();
                    for (size_t i = 0; i < shape_.elements(); ++i)
                    {
                        dst[i] = static_cast<double>(src[i]);
                    }
                    break;
                }
                case DataType::kInt32:
                {
                    int32_t *dst = result->data_ptr<int32_t>();
                    for (size_t i = 0; i < shape_.elements(); ++i)
                    {
                        dst[i] = static_cast<int32_t>(src[i]);
                    }
                    break;
                }
                case DataType::kInt8:
                {
                    int8_t *dst = result->data_ptr<int8_t>();
                    for (size_t i = 0; i < shape_.elements(); ++i)
                    {
                        dst[i] = static_cast<int8_t>(src[i]);
                    }
                    break;
                }
                default:
                    throw std::invalid_argument("Unsupported target type for conversion");
            }
            break;
        }
        case DataType::kFloat64:
        {
            const double *src = data_ptr<double>();
            switch (target_type)
            {
                case DataType::kFloat32:
                {
                    float *dst = result->data_ptr<float>();
                    for (size_t i = 0; i < shape_.elements(); ++i)
                    {
                        dst[i] = static_cast<float>(src[i]);
                    }
                    break;
                }
                case DataType::kInt32:
                {
                    int32_t *dst = result->data_ptr<int32_t>();
                    for (size_t i = 0; i < shape_.elements(); ++i)
                    {
                        dst[i] = static_cast<int32_t>(src[i]);
                    }
                    break;
                }
                case DataType::kInt8:
                {
                    int8_t *dst = result->data_ptr<int8_t>();
                    for (size_t i = 0; i < shape_.elements(); ++i)
                    {
                        dst[i] = static_cast<int8_t>(src[i]);
                    }
                    break;
                }
                default:
                    throw std::invalid_argument("Unsupported target type for conversion");
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported source type for conversion");
    }

    return result;
}

Device OriginMat::device() const
{
    return Device(storage_->device_type(), storage_->device_index());
}

std::unique_ptr<Mat> OriginMat::to_device(Device device) const
{
    if (device == this->device())
    {
        return clone();
    }

    auto new_storage = storage_->to_device(device.type(), device.index());
    return std::make_unique<OriginMat>(new_storage, shape_, dtype_);
}

// 可视化
void OriginMat::print(const std::string &desc) const
{
    auto data_vec          = to_vector();
    auto shape_vec         = shape_.dims();
    std::string device_str = device().to_string();

    utils::visualize::print_origin_mat(desc, data_vec, shape_vec, dtype_, device_str);
}

int OriginMat::backend_type() const
{
    return 2;  // ORIGIN backend
}

// 工厂方法
std::unique_ptr<Mat> OriginMat::randn(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    std::random_device rd;
    std::mt19937 gen(rd());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            std::normal_distribution<float> dist(0.0f, 1.0f);
            float *data = result->data_ptr<float>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = dist(gen);
            }
            break;
        }
        case DataType::kFloat64:
        {
            std::normal_distribution<double> dist(0.0, 1.0);
            double *data = result->data_ptr<double>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = dist(gen);
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for randn");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::zeros(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float *data = result->data_ptr<float>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0.0f;
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = result->data_ptr<double>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0.0;
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0;
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 0;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for zeros");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::ones(const Shape &shape, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float *data = result->data_ptr<float>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1.0f;
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = result->data_ptr<double>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1.0;
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = result->data_ptr<int32_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1;
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = result->data_ptr<int8_t>();
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = 1;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for ones");
    }

    return result;
}

std::unique_ptr<Mat> OriginMat::full(const Shape &shape, data_t value, const TensorOptions &options)
{
    auto result = std::make_unique<OriginMat>(shape, options.dtype());

    switch (options.dtype())
    {
        case DataType::kFloat32:
        {
            float *data = result->data_ptr<float>();
            float val   = static_cast<float>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = val;
            }
            break;
        }
        case DataType::kFloat64:
        {
            double *data = result->data_ptr<double>();
            double val   = static_cast<double>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = val;
            }
            break;
        }
        case DataType::kInt32:
        {
            int32_t *data = result->data_ptr<int32_t>();
            int32_t val   = static_cast<int32_t>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = val;
            }
            break;
        }
        case DataType::kInt8:
        {
            int8_t *data = result->data_ptr<int8_t>();
            int8_t val   = static_cast<int8_t>(value);
            for (size_t i = 0; i < shape.elements(); ++i)
            {
                data[i] = val;
            }
            break;
        }
        default:
            throw std::invalid_argument("Unsupported data type for full");
    }

    return result;
}

// 显式实例化模板构造函数
template OriginMat::OriginMat(const std::vector<float> &, const Shape &);
template OriginMat::OriginMat(const std::vector<double> &, const Shape &);
template OriginMat::OriginMat(const std::vector<int32_t> &, const Shape &);
template OriginMat::OriginMat(const std::vector<int8_t> &, const Shape &);

template OriginMat::OriginMat(float, const Shape &);
template OriginMat::OriginMat(double, const Shape &);
template OriginMat::OriginMat(int32_t, const Shape &);
template OriginMat::OriginMat(int8_t, const Shape &);

}  // namespace origin
