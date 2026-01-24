#include "origin/mat/torch/torch_mat.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include "origin/utils/exception.h"

namespace origin
{

namespace
{

// 映射 OriginDL 的 DataType 到 LibTorch 的 ScalarType
at::ScalarType to_torch_scalar_type(DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return at::kFloat;
        case DataType::kFloat64:
            return at::kDouble;
        case DataType::kInt8:
            return at::kChar;
        case DataType::kInt32:
            return at::kInt;
        case DataType::kInt64:
            return at::kLong;
        case DataType::kUInt8:
            return at::kByte;
        default:
            THROW_INVALID_ARG("Unsupported DataType {} for TorchMat", static_cast<int>(dtype));
    }
}

DataType from_torch_scalar_type(at::ScalarType scalar_type)
{
    switch (scalar_type)
    {
        case at::kFloat:
            return DataType::kFloat32;
        case at::kDouble:
            return DataType::kFloat64;
        case at::kChar:
            return DataType::kInt8;
        case at::kInt:
            return DataType::kInt32;
        case at::kLong:
            return DataType::kInt64;
        case at::kByte:
            return DataType::kUInt8;
        default:
            THROW_INVALID_ARG("Unsupported Torch scalar type {}", static_cast<int>(scalar_type));
    }
}

torch::Device to_torch_device(const Device &device)
{
    if (device.type() == DeviceType::kCPU)
    {
        return torch::Device(torch::kCPU);
    }
    if (device.type() == DeviceType::kCUDA)
    {
        return torch::Device(torch::kCUDA, device.index());
    }
    THROW_INVALID_ARG("Unsupported device type {}", static_cast<int>(device.type()));
}

Device from_torch_device(const torch::Device &device)
{
    if (device.type() == torch::kCPU)
    {
        return Device(DeviceType::kCPU, 0);
    }
    if (device.type() == torch::kCUDA)
    {
        return Device(DeviceType::kCUDA, device.index());
    }
    THROW_INVALID_ARG("Unsupported Torch device type {}", static_cast<int>(device.type()));
}

std::vector<int64_t> shape_to_sizes(const Shape &shape)
{
    const auto &dims = shape.dims();
    std::vector<int64_t> sizes;
    sizes.reserve(dims.size());
    for (auto d : dims)
    {
        sizes.push_back(static_cast<int64_t>(d));
    }
    return sizes;
}

Shape sizes_to_shape(const c10::IntArrayRef &sizes)
{
    std::vector<size_t> dims;
    dims.reserve(sizes.size());
    for (auto s : sizes)
    {
        dims.push_back(static_cast<size_t>(s));
    }
    return Shape(dims);
}

torch::TensorOptions make_torch_options(const TensorOptions &options)
{
    auto scalar_type  = to_torch_scalar_type(options.dtype());
    auto torch_device = to_torch_device(options.device());
    torch::TensorOptions torch_opts;
    torch_opts = torch_opts.dtype(scalar_type).device(torch_device);
    // 不使用 LibTorch 的 requires_grad，梯度由 OriginDL 自己管理
    return torch_opts;
}

at::Scalar make_torch_scalar(const Scalar &scalar, DataType target_dtype)
{
    switch (target_dtype)
    {
        case DataType::kFloat32:
            return at::Scalar(scalar.to<float>());
        case DataType::kFloat64:
            return at::Scalar(scalar.to<double>());
        case DataType::kInt8:
            return at::Scalar(scalar.to<int8_t>());
        case DataType::kInt32:
            return at::Scalar(scalar.to<int32_t>());
        case DataType::kInt64:
            return at::Scalar(scalar.to<int64_t>());
        case DataType::kUInt8:
            return at::Scalar(scalar.to<uint8_t>());
        default:
            THROW_INVALID_ARG("Unsupported DataType {} for scalar conversion", static_cast<int>(target_dtype));
    }
}

// 将 Mat 引用安全地转换为 TorchMat 引用
const TorchMat &as_torch_mat(const Mat &m, const char *op_name)
{
    auto ptr = dynamic_cast<const TorchMat *>(&m);
    if (!ptr)
    {
        THROW_RUNTIME_ERROR("{}: Mat backend must be TorchMat (backend_type={} ), got {}", op_name, TORCH_BACKEND_TYPE,
                            m.backend_type());
    }
    return *ptr;
}

TorchMat &as_torch_mat(Mat &m, const char *op_name)
{
    auto ptr = dynamic_cast<TorchMat *>(&m);
    if (!ptr)
    {
        THROW_RUNTIME_ERROR("{}: Mat backend must be TorchMat (backend_type={} ), got {}", op_name, TORCH_BACKEND_TYPE,
                            m.backend_type());
    }
    return *ptr;
}

// 计算按行优先（row-major）的扁平索引，假设张量是连续的
size_t compute_flat_index(const Shape &shape, std::initializer_list<size_t> indices)
{
    if (indices.size() != shape.size())
    {
        THROW_INVALID_ARG("Index count ({}) does not match tensor dimension ({}). Shape: {}", indices.size(),
                          shape.size(), shape.to_string());
    }

    size_t flat_index = 0;
    size_t stride     = 1;
    size_t dim_count  = shape.size();

    // 从最后一维开始计算
    size_t idx_pos = dim_count;
    auto it        = indices.end();
    while (idx_pos > 0)
    {
        --idx_pos;
        --it;
        size_t idx = *it;
        if (idx >= shape[idx_pos])
        {
            THROW_INVALID_ARG("Index {} out of range for dimension {} (size: {}). Shape: {}", idx, idx_pos,
                              shape[idx_pos], shape.to_string());
        }
        flat_index += idx * stride;
        stride *= shape[idx_pos];
    }

    return flat_index;
}

}  // namespace

// 构造函数
TorchMat::TorchMat(const torch::Tensor &tensor) : tensor_(tensor) {}

TorchMat::TorchMat(torch::Tensor &&tensor) : tensor_(std::move(tensor)) {}

// === 静态工厂方法实现 ===

std::unique_ptr<Mat> TorchMat::from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
{
    auto sizes      = shape_to_sizes(shape);
    auto torch_opts = make_torch_options(options);
    auto torch_val  = make_torch_scalar(scalar, options.dtype());

    auto t = torch::full(sizes, torch_val, torch_opts);
    return std::make_unique<TorchMat>(std::move(t));
}

std::unique_ptr<Mat> TorchMat::from_memory(const void *data,
                                           DataType user_dtype,
                                           const Shape &shape,
                                           const TensorOptions &options)
{
    auto sizes = shape_to_sizes(shape);

    // 先在 CPU 上按照用户数据类型创建张量视图，然后 clone 拷贝一份由 Torch 持有的内存
    auto src_opts = torch::TensorOptions().dtype(to_torch_scalar_type(user_dtype)).device(torch::kCPU);

    auto src = torch::from_blob(const_cast<void *>(data), sizes, src_opts).clone();

    // 数据类型转换
    if (options.dtype() != user_dtype)
    {
        src = src.to(to_torch_scalar_type(options.dtype()));
    }

    // 设备转换（当前绝大多数测试都在 CPU 上，这里逻辑先保留）
    if (options.device().type() == DeviceType::kCUDA)
    {
        src = src.to(to_torch_device(options.device()));
    }

    return std::make_unique<TorchMat>(std::move(src));
}

std::unique_ptr<Mat> TorchMat::randn(const Shape &shape, const TensorOptions &options)
{
    auto sizes      = shape_to_sizes(shape);
    auto torch_opts = make_torch_options(options);
    auto t          = torch::randn(sizes, torch_opts);
    return std::make_unique<TorchMat>(std::move(t));
}

// === 卷积 / 池化 / BatchNorm / 其他复杂算子 ===
// 这些接口当前仍然未在 Torch 后端实现，需要时可以基于 LibTorch 逐步补全。

std::unique_ptr<Mat> TorchMat::im2col(std::pair<int, int>, std::pair<int, int>, std::pair<int, int>, bool) const
{
    THROW_RUNTIME_ERROR("TorchMat::im2col is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::col2im(const Shape &,
                                      std::pair<int, int>,
                                      std::pair<int, int>,
                                      std::pair<int, int>,
                                      bool) const
{
    THROW_RUNTIME_ERROR("TorchMat::col2im is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::conv2d(const Mat &, const Mat *, std::pair<int, int>, std::pair<int, int>) const
{
    THROW_RUNTIME_ERROR("TorchMat::conv2d is not implemented yet. Please use OriginMat backend.");
}

std::vector<std::unique_ptr<Mat>> TorchMat::conv2d_backward(const Mat &,
                                                            const Mat &,
                                                            const Mat &,
                                                            const Mat *,
                                                            std::pair<int, int>,
                                                            std::pair<int, int>) const
{
    THROW_RUNTIME_ERROR("TorchMat::conv2d_backward is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::avg_pool2d(std::pair<int, int>, std::pair<int, int>, std::pair<int, int>) const
{
    THROW_RUNTIME_ERROR("TorchMat::avg_pool2d is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::avg_pool2d_backward(const Mat &,
                                                   std::pair<int, int>,
                                                   std::pair<int, int>,
                                                   std::pair<int, int>) const
{
    THROW_RUNTIME_ERROR("TorchMat::avg_pool2d_backward is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::adaptive_avg_pool2d(std::pair<int, int>) const
{
    THROW_RUNTIME_ERROR("TorchMat::adaptive_avg_pool2d is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::adaptive_avg_pool2d_backward(const Mat &, std::pair<int, int>) const
{
    THROW_RUNTIME_ERROR("TorchMat::adaptive_avg_pool2d_backward is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::max_pool2d(std::pair<int, int>,
                                          std::pair<int, int>,
                                          std::pair<int, int>,
                                          std::vector<size_t> &) const
{
    THROW_RUNTIME_ERROR("TorchMat::max_pool2d is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::max_pool2d_backward(const Mat &,
                                                   std::pair<int, int>,
                                                   std::pair<int, int>,
                                                   std::pair<int, int>,
                                                   const std::vector<size_t> &) const
{
    THROW_RUNTIME_ERROR("TorchMat::max_pool2d_backward is not implemented yet. Please use OriginMat backend.");
}

Mat::BatchNormResult TorchMat::batch_norm_forward(const Mat &, const Mat &, const Mat &, const Mat &, bool, float, int)
    const
{
    THROW_RUNTIME_ERROR("TorchMat::batch_norm_forward is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::batch_norm(const Mat &, const Mat &, const Mat &, const Mat &, bool, float, float, int)
    const
{
    THROW_RUNTIME_ERROR("TorchMat::batch_norm is not implemented yet. Please use OriginMat backend.");
}

std::vector<std::unique_ptr<Mat>>
TorchMat::batch_norm_backward(const Mat &, const Mat &, const Mat &, const Mat &, const Mat &, float, int) const
{
    THROW_RUNTIME_ERROR("TorchMat::batch_norm_backward is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::gather(const Mat &) const
{
    THROW_RUNTIME_ERROR("TorchMat::gather is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::one_hot(const Mat &, int) const
{
    THROW_RUNTIME_ERROR("TorchMat::one_hot is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat>
TorchMat::yolo_detect_forward(const Mat &, const Mat *, const Mat &, const Mat &, float, int32_t, int32_t) const
{
    THROW_RUNTIME_ERROR("TorchMat::yolo_detect_forward is not implemented yet. Please use OriginMat backend.");
}

// === 基本张量操作实现 ===

std::unique_ptr<Mat> TorchMat::clone() const
{
    return std::make_unique<TorchMat>(tensor_.clone());
}

std::unique_ptr<Mat> TorchMat::view(const Shape &new_shape) const
{
    if (!tensor_.is_contiguous())
    {
        THROW_RUNTIME_ERROR("TorchMat::view requires contiguous tensor");
    }

    if (new_shape.elements() != elements())
    {
        THROW_INVALID_ARG("TorchMat::view: total elements must match. Original: {}, Target: {}", elements(),
                          new_shape.elements());
    }

    auto sizes = shape_to_sizes(new_shape);
    auto t     = tensor_.view(sizes);
    return std::make_unique<TorchMat>(std::move(t));
}

bool TorchMat::is_contiguous() const
{
    return tensor_.is_contiguous();
}

std::unique_ptr<Mat> TorchMat::contiguous() const
{
    if (tensor_.is_contiguous())
    {
        return clone();
    }
    auto t = tensor_.contiguous();
    return std::make_unique<TorchMat>(std::move(t));
}

std::unique_ptr<Mat> TorchMat::reshape(const Shape &new_shape) const
{
    if (new_shape.elements() != elements())
    {
        THROW_INVALID_ARG("TorchMat::reshape: total elements must match. Original: {}, Target: {}", elements(),
                          new_shape.elements());
    }

    auto sizes = shape_to_sizes(new_shape);
    auto t     = tensor_.reshape(sizes);
    return std::make_unique<TorchMat>(std::move(t));
}

std::unique_ptr<Mat> TorchMat::transpose() const
{
    // 与 OriginMat 一致：做数据转置（产生新 tensor），而不是视图转置
    int64_t dim0 = 0;
    int64_t dim1 = 0;

    if (tensor_.dim() < 2)
    {
        return clone();
    }

    if (tensor_.dim() == 2)
    {
        dim0 = 0;
        dim1 = 1;
    }
    else
    {
        dim0 = tensor_.dim() - 2;
        dim1 = tensor_.dim() - 1;
    }

    auto t = tensor_.transpose(dim0, dim1).contiguous();
    return std::make_unique<TorchMat>(std::move(t));
}

// === 二元运算（加减乘除 + matmul）===

std::unique_ptr<Mat> TorchMat::operator+(const Mat &other) const
{
    const auto &rhs = as_torch_mat(other, "TorchMat::operator+");
    auto t          = tensor_ + rhs.tensor_;
    return std::make_unique<TorchMat>(std::move(t));
}

void TorchMat::add_inplace(const Mat &other)
{
    const auto &rhs = as_torch_mat(other, "TorchMat::add_inplace");
    tensor_.add_(rhs.tensor_);
}

std::unique_ptr<Mat> TorchMat::operator-(const Mat &other) const
{
    const auto &rhs = as_torch_mat(other, "TorchMat::operator-");
    auto t          = tensor_ - rhs.tensor_;
    return std::make_unique<TorchMat>(std::move(t));
}

void TorchMat::sub_inplace(const Mat &other)
{
    const auto &rhs = as_torch_mat(other, "TorchMat::sub_inplace");
    tensor_.sub_(rhs.tensor_);
}

std::unique_ptr<Mat> TorchMat::operator*(const Mat &other) const
{
    const auto &rhs = as_torch_mat(other, "TorchMat::operator*");
    auto t          = tensor_ * rhs.tensor_;
    return std::make_unique<TorchMat>(std::move(t));
}

void TorchMat::mul_inplace(const Mat &other)
{
    const auto &rhs = as_torch_mat(other, "TorchMat::mul_inplace");
    tensor_.mul_(rhs.tensor_);
}

std::unique_ptr<Mat> TorchMat::matmul(const Mat &other) const
{
    const auto &rhs = as_torch_mat(other, "TorchMat::matmul");
    auto t          = torch::matmul(tensor_, rhs.tensor_);
    return std::make_unique<TorchMat>(std::move(t));
}

std::unique_ptr<Mat> TorchMat::operator/(const Mat &other) const
{
    const auto &rhs = as_torch_mat(other, "TorchMat::operator/");
    auto t          = tensor_ / rhs.tensor_;
    return std::make_unique<TorchMat>(std::move(t));
}

void TorchMat::div_inplace(const Mat &other)
{
    const auto &rhs = as_torch_mat(other, "TorchMat::div_inplace");
    tensor_.div_(rhs.tensor_);
}

std::unique_ptr<Mat> TorchMat::operator-() const
{
    auto t = -tensor_;
    return std::make_unique<TorchMat>(std::move(t));
}

// === 广播与归约 ===

std::unique_ptr<Mat> TorchMat::broadcast_to(const Shape &target_shape) const
{
    auto target_sizes = shape_to_sizes(target_shape);
    auto t            = tensor_.expand(target_sizes).clone();
    return std::make_unique<TorchMat>(std::move(t));
}

std::unique_ptr<Mat> TorchMat::sum_to(const Shape &target_shape) const
{
    Shape src_shape = shape();

    if (src_shape == target_shape)
    {
        return clone();
    }

    size_t current_elements = elements();
    size_t target_elements  = target_shape.elements();

    if (target_elements > current_elements)
    {
        THROW_RUNTIME_ERROR("TorchMat::sum_to: target elements ({}) cannot be larger than source ({})", target_elements,
                            current_elements);
    }

    const auto &src_dims = src_shape.dims();
    const auto &tgt_dims = target_shape.dims();

    std::vector<int64_t> sum_dims;
    size_t min_dims = std::min(src_dims.size(), tgt_dims.size());

    for (size_t i = 0; i < min_dims; ++i)
    {
        if (tgt_dims[i] == 1 && src_dims[i] > 1)
        {
            sum_dims.push_back(static_cast<int64_t>(i));
        }
    }

    if (src_dims.size() > tgt_dims.size())
    {
        for (size_t i = tgt_dims.size(); i < src_dims.size(); ++i)
        {
            sum_dims.push_back(static_cast<int64_t>(i));
        }
    }

    auto result = tensor_;

    std::sort(sum_dims.begin(), sum_dims.end(), std::greater<int64_t>());
    for (auto dim : sum_dims)
    {
        result = result.sum(dim, /*keepdim=*/false);
    }

    if (sizes_to_shape(result.sizes()) != target_shape)
    {
        auto sizes = shape_to_sizes(target_shape);
        result     = result.reshape(sizes);
    }

    return std::make_unique<TorchMat>(std::move(result));
}

std::unique_ptr<Mat> TorchMat::sum(int axis) const
{
    torch::Tensor result;
    if (axis == -1)
    {
        result = tensor_.sum();
    }
    else
    {
        result = tensor_.sum(axis);
    }
    return std::make_unique<TorchMat>(std::move(result));
}

std::unique_ptr<Mat> TorchMat::max(int axis) const
{
    torch::Tensor result;
    if (axis == -1)
    {
        result = std::get<0>(tensor_.max(/*dim=*/0, /*keepdim=*/false));
        for (int64_t d = 1; d < tensor_.dim(); ++d)
        {
            result = std::get<0>(result.max(/*dim=*/0, /*keepdim=*/false));
        }
    }
    else
    {
        result = std::get<0>(tensor_.max(axis));
    }
    return std::make_unique<TorchMat>(std::move(result));
}

// 数学函数（本任务暂不需要，先保留抛异常实现，未来可基于 LibTorch 完成）

std::unique_ptr<Mat> TorchMat::exp() const
{
    THROW_RUNTIME_ERROR("TorchMat::exp is not implemented yet.");
}

void TorchMat::exp_inplace()
{
    THROW_RUNTIME_ERROR("TorchMat::exp_inplace is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::log() const
{
    THROW_RUNTIME_ERROR("TorchMat::log is not implemented yet.");
}

void TorchMat::log_inplace()
{
    THROW_RUNTIME_ERROR("TorchMat::log_inplace is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::sin() const
{
    THROW_RUNTIME_ERROR("TorchMat::sin is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::cos() const
{
    THROW_RUNTIME_ERROR("TorchMat::cos is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::sqrt() const
{
    THROW_RUNTIME_ERROR("TorchMat::sqrt is not implemented yet.");
}

void TorchMat::sqrt_inplace()
{
    THROW_RUNTIME_ERROR("TorchMat::sqrt_inplace is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::square() const
{
    THROW_RUNTIME_ERROR("TorchMat::square is not implemented yet.");
}

void TorchMat::square_inplace()
{
    THROW_RUNTIME_ERROR("TorchMat::square_inplace is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::pow(const Scalar &) const
{
    THROW_RUNTIME_ERROR("TorchMat::pow is not implemented yet.");
}

void TorchMat::pow_inplace(const Scalar &)
{
    THROW_RUNTIME_ERROR("TorchMat::pow_inplace is not implemented yet.");
}

std::unique_ptr<Mat> TorchMat::relu() const
{
    THROW_RUNTIME_ERROR("TorchMat::relu is not implemented yet.");
}

void TorchMat::relu_inplace()
{
    THROW_RUNTIME_ERROR("TorchMat::relu_inplace is not implemented yet.");
}

void TorchMat::neg_inplace()
{
    tensor_.neg_();
}

// === 形状和属性 ===

Shape TorchMat::shape() const
{
    return sizes_to_shape(tensor_.sizes());
}

size_t TorchMat::elements() const
{
    return static_cast<size_t>(tensor_.numel());
}

bool TorchMat::is_scalar() const
{
    return tensor_.dim() == 0;
}

Scalar TorchMat::scalar_value() const
{
    if (elements() != 1)
    {
        THROW_INVALID_ARG("scalar_value() can only be called on scalar tensors, but tensor has {} elements",
                          elements());
    }

    switch (dtype())
    {
        case DataType::kFloat32:
            return Scalar(tensor_.item<float>());
        case DataType::kFloat64:
            return Scalar(tensor_.item<double>());
        case DataType::kInt8:
            return Scalar(tensor_.item<int8_t>());
        case DataType::kInt32:
            return Scalar(tensor_.item<int32_t>());
        case DataType::kInt64:
            return Scalar(tensor_.item<int64_t>());
        case DataType::kUInt8:
            return Scalar(tensor_.item<uint8_t>());
        default:
            THROW_INVALID_ARG("Unsupported dtype {} in scalar_value()", static_cast<int>(dtype()));
    }
}

Scalar TorchMat::index(std::initializer_list<size_t> indices) const
{
    Shape s           = shape();
    size_t flat_index = compute_flat_index(s, indices);

    switch (dtype())
    {
        case DataType::kFloat32:
        {
            const float *p = tensor_.data_ptr<float>();
            return Scalar(p[flat_index]);
        }
        case DataType::kFloat64:
        {
            const double *p = tensor_.data_ptr<double>();
            return Scalar(p[flat_index]);
        }
        case DataType::kInt8:
        {
            const int8_t *p = tensor_.data_ptr<int8_t>();
            return Scalar(p[flat_index]);
        }
        case DataType::kInt32:
        {
            const int32_t *p = tensor_.data_ptr<int32_t>();
            return Scalar(p[flat_index]);
        }
        case DataType::kInt64:
        {
            const int64_t *p = tensor_.data_ptr<int64_t>();
            return Scalar(p[flat_index]);
        }
        case DataType::kUInt8:
        {
            const uint8_t *p = tensor_.data_ptr<uint8_t>();
            return Scalar(p[flat_index]);
        }
        default:
            THROW_INVALID_ARG("Unsupported dtype {} in index()", static_cast<int>(dtype()));
    }
}

void TorchMat::index_put(std::initializer_list<size_t> indices, const Scalar &value)
{
    Shape s           = shape();
    size_t flat_index = compute_flat_index(s, indices);

    switch (dtype())
    {
        case DataType::kFloat32:
        {
            float *p      = tensor_.data_ptr<float>();
            p[flat_index] = value.to<float>();
            break;
        }
        case DataType::kFloat64:
        {
            double *p     = tensor_.data_ptr<double>();
            p[flat_index] = value.to<double>();
            break;
        }
        case DataType::kInt8:
        {
            int8_t *p     = tensor_.data_ptr<int8_t>();
            p[flat_index] = value.to<int8_t>();
            break;
        }
        case DataType::kInt32:
        {
            int32_t *p    = tensor_.data_ptr<int32_t>();
            p[flat_index] = value.to<int32_t>();
            break;
        }
        case DataType::kInt64:
        {
            int64_t *p    = tensor_.data_ptr<int64_t>();
            p[flat_index] = value.to<int64_t>();
            break;
        }
        case DataType::kUInt8:
        {
            uint8_t *p    = tensor_.data_ptr<uint8_t>();
            p[flat_index] = value.to<uint8_t>();
            break;
        }
        default:
            THROW_INVALID_ARG("Unsupported dtype {} in index_put()", static_cast<int>(dtype()));
    }
}

void *TorchMat::data_ptr()
{
    if (!tensor_.is_contiguous())
    {
        THROW_RUNTIME_ERROR("TorchMat::data_ptr currently only supports contiguous tensors");
    }
    // 返回基础数据指针，由上层通过模板 data_ptr<T>() 进行类型转换
    return tensor_.data_ptr();
}

void TorchMat::print(const std::string &desc) const
{
    if (!desc.empty())
    {
        std::cout << desc << std::endl;
    }
    std::cout << tensor_ << std::endl;
}

std::vector<float> TorchMat::to_vector() const
{
    torch::Tensor t = tensor_;

    // 确保在 CPU 上
    if (t.device().type() != torch::kCPU)
    {
        t = t.to(torch::kCPU);
    }

    // 转为 float32
    if (t.scalar_type() != at::kFloat)
    {
        t = t.to(at::kFloat);
    }

    t = t.contiguous();

    size_t n = static_cast<size_t>(t.numel());
    std::vector<float> v(n);
    std::memcpy(v.data(), t.data_ptr<float>(), n * sizeof(float));
    return v;
}

// === 类型和设备 ===

int TorchMat::backend_type() const
{
    return TORCH_BACKEND_TYPE;
}

DataType TorchMat::dtype() const
{
    return from_torch_scalar_type(tensor_.scalar_type());
}

Device TorchMat::device() const
{
    return from_torch_device(tensor_.device());
}

std::unique_ptr<Mat> TorchMat::to(DataType target_type) const
{
    auto t = tensor_.to(to_torch_scalar_type(target_type));
    return std::make_unique<TorchMat>(std::move(t));
}

std::unique_ptr<Mat> TorchMat::to_device(Device device) const
{
    auto t = tensor_.to(to_torch_device(device));
    return std::make_unique<TorchMat>(std::move(t));
}

// === Dropout / Upsample 相关（暂未在 Torch 后端实现）===

std::unique_ptr<Mat> TorchMat::dropout(float, bool, Mat *) const
{
    THROW_RUNTIME_ERROR("TorchMat::dropout is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::dropout_backward(const Mat &, const Mat &) const
{
    THROW_RUNTIME_ERROR("TorchMat::dropout_backward is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::upsample(const Shape &, int, int) const
{
    THROW_RUNTIME_ERROR("TorchMat::upsample is not implemented yet. Please use OriginMat backend.");
}

std::unique_ptr<Mat> TorchMat::upsample_backward(const Mat &, const Shape &, int, int) const
{
    THROW_RUNTIME_ERROR("TorchMat::upsample_backward is not implemented yet. Please use OriginMat backend.");
}

}  // namespace origin
