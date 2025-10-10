#include "origin/mat/torch/torch_mat.h"
#include <torch/torch.h>
#include <stdexcept>
#include "origin/mat/backend_constants.h"
#include "origin/utils/log.h"

namespace origin
{

TorchMat::TorchMat(const std::vector<data_t> &data, const Shape &shape)
{
    // 验证数据是否为空
    if (data.empty())
    {
        throw std::invalid_argument("TorchMat: Tensor data cannot be empty. Data vector is empty.");
    }

    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("TorchMat: Tensor shape cannot have zero dimensions. Dimension " +
                                        std::to_string(i) + " is zero in shape " + shape.to_string());
        }
    }

    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    data_ = torch::from_blob(const_cast<data_t*>(data.data()), sizes, torch::kFloat32).clone();
}

TorchMat::TorchMat(data_t value, const Shape &shape)
{
    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("TorchMat: Tensor shape cannot have zero dimensions. Dimension " +
                                        std::to_string(i) + " is zero in shape " + shape.to_string());
        }
    }

    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    data_ = torch::full(sizes, value, torch::kFloat32);
}

std::unique_ptr<Mat> TorchMat::clone() const
{
    return std::make_unique<TorchMat>(data_.clone());
}

std::unique_ptr<Mat> TorchMat::reshape(const Shape &shape) const
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    return std::make_unique<TorchMat>(data_.reshape(sizes));
}

std::unique_ptr<Mat> TorchMat::transpose() const
{
    return std::make_unique<TorchMat>(data_.transpose(-2, -1));
}

std::unique_ptr<Mat> TorchMat::operator+(const Mat &other) const
{
    const TorchMat &other_torch = dynamic_cast<const TorchMat &>(other);
    return std::make_unique<TorchMat>(data_ + other_torch.data_);
}

std::unique_ptr<Mat> TorchMat::operator-(const Mat &other) const
{
    const TorchMat &other_torch = dynamic_cast<const TorchMat &>(other);
    return std::make_unique<TorchMat>(data_ - other_torch.data_);
}

std::unique_ptr<Mat> TorchMat::operator*(const Mat &other) const
{
    const TorchMat &other_torch = dynamic_cast<const TorchMat &>(other);
    return std::make_unique<TorchMat>(data_ * other_torch.data_);
}

std::unique_ptr<Mat> TorchMat::matmul(const Mat &other) const
{
    const TorchMat &other_torch = dynamic_cast<const TorchMat &>(other);
    return std::make_unique<TorchMat>(torch::matmul(data_, other_torch.data_));
}

std::unique_ptr<Mat> TorchMat::operator/(const Mat &other) const
{
    const TorchMat &other_torch = dynamic_cast<const TorchMat &>(other);
    return std::make_unique<TorchMat>(data_ / other_torch.data_);
}

std::unique_ptr<Mat> TorchMat::add_scalar(data_t scalar) const
{
    return std::make_unique<TorchMat>(data_ + scalar);
}

std::unique_ptr<Mat> TorchMat::mul_scalar(data_t scalar) const
{
    return std::make_unique<TorchMat>(data_ * scalar);
}

std::unique_ptr<Mat> TorchMat::operator+(data_t scalar) const
{
    return std::make_unique<TorchMat>(data_ + scalar);
}

std::unique_ptr<Mat> TorchMat::operator-(data_t scalar) const
{
    return std::make_unique<TorchMat>(data_ - scalar);
}

std::unique_ptr<Mat> TorchMat::operator*(data_t scalar) const
{
    return std::make_unique<TorchMat>(data_ * scalar);
}

std::unique_ptr<Mat> TorchMat::operator/(data_t scalar) const
{
    return std::make_unique<TorchMat>(data_ / scalar);
}

std::unique_ptr<Mat> TorchMat::operator-() const
{
    return std::make_unique<TorchMat>(-data_);
}

std::unique_ptr<Mat> TorchMat::broadcast_to(const Shape &shape) const
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    return std::make_unique<TorchMat>(data_.expand(sizes));
}

std::unique_ptr<Mat> TorchMat::sum_to(const Shape &shape) const
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    auto result = data_;
    
    // 计算需要求和的维度
    auto current_sizes = data_.sizes();
    auto target_sizes = sizes;
    
    // 从右到左处理维度
    for (int i = current_sizes.size() - 1, j = target_sizes.size() - 1; i >= 0 && j >= 0; --i, --j)
    {
        if (current_sizes[i] != target_sizes[j])
        {
            result = result.sum(i, true);
        }
    }
    
    // 处理多余的维度
    for (int i = 0; i < current_sizes.size() - target_sizes.size(); ++i)
    {
        result = result.sum(0, true);
    }
    
    return std::make_unique<TorchMat>(result.reshape(sizes));
}

std::unique_ptr<Mat> TorchMat::sum(int axis) const
{
    if (axis == -1)
    {
        // 对所有元素求和，返回标量
        auto result = data_.sum();
        return std::make_unique<TorchMat>(result);
    }
    else
    {
        // 沿指定轴求和，保持维度
        return std::make_unique<TorchMat>(data_.sum(axis, true));
    }
}

Shape TorchMat::shape() const
{
    return TorchMat::convert_torch_sizes_to_shape(data_.sizes());
}

size_t TorchMat::elements() const
{
    return data_.numel();
}

std::vector<data_t> TorchMat::to_vector() const
{
    return TorchMat::tensor_to_vector(data_);
}

// 数学函数实现
std::unique_ptr<Mat> TorchMat::exp() const
{
    return std::make_unique<TorchMat>(torch::exp(data_));
}

std::unique_ptr<Mat> TorchMat::log() const
{
    return std::make_unique<TorchMat>(torch::log(data_));
}

std::unique_ptr<Mat> TorchMat::sin() const
{
    return std::make_unique<TorchMat>(torch::sin(data_));
}

std::unique_ptr<Mat> TorchMat::cos() const
{
    return std::make_unique<TorchMat>(torch::cos(data_));
}

std::unique_ptr<Mat> TorchMat::sqrt() const
{
    return std::make_unique<TorchMat>(torch::sqrt(data_));
}

std::unique_ptr<Mat> TorchMat::square() const
{
    return std::make_unique<TorchMat>(data_ * data_);
}

std::unique_ptr<Mat> TorchMat::pow(data_t exponent) const
{
    return std::make_unique<TorchMat>(torch::pow(data_, exponent));
}

// 数据访问方法
template <typename T>
T TorchMat::scalar() const
{
    return data_.item<T>();
}

// 调试方法
void TorchMat::print(const std::string &desc) const
{
    std::cout << "DL Mat Shape: " << shape() << std::endl;
    if (!desc.empty())
    {
        std::cout << desc << ": " << std::endl;
    }
    std::cout << data_ << std::endl;
}

// 显式实例化
template data_t TorchMat::scalar<data_t>() const;
template int TorchMat::scalar<int>() const;

data_t TorchMat::sum() const
{
    return data_.sum().item<data_t>();
}

data_t TorchMat::max() const
{
    return data_.max().item<data_t>();
}

data_t TorchMat::min() const
{
    return data_.min().item<data_t>();
}

data_t TorchMat::mean() const
{
    return data_.mean().item<data_t>();
}

int TorchMat::backend_type() const
{
    return TORCH_CONST;
}

// 静态辅助函数实现
std::vector<data_t> TorchMat::tensor_to_vector(const torch::Tensor &tensor)
{
    std::vector<data_t> result(tensor.numel());
    auto data_ptr = tensor.data_ptr<data_t>();
    std::copy(data_ptr, data_ptr + tensor.numel(), result.begin());
    return result;
}

torch::Tensor TorchMat::vector_to_tensor(const std::vector<data_t> &data, const Shape &shape)
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    return torch::from_blob(const_cast<data_t*>(data.data()), sizes, torch::kFloat32).clone();
}

std::vector<int64_t> TorchMat::convert_shape_to_torch_sizes(const Shape &shape)
{
    const auto &dims = shape.dims();
    if (dims.empty())
    {
        return {1};
    }
    
    std::vector<int64_t> sizes;
    sizes.reserve(dims.size());
    for (size_t dim : dims)
    {
        sizes.push_back(static_cast<int64_t>(dim));
    }
    
    return sizes;
}

Shape TorchMat::convert_torch_sizes_to_shape(const torch::IntArrayRef &sizes)
{
    std::vector<size_t> shape_dims;
    shape_dims.reserve(sizes.size());
    for (int64_t size : sizes)
    {
        shape_dims.push_back(static_cast<size_t>(size));
    }
    return Shape(shape_dims);
}

// 静态工厂方法实现
std::unique_ptr<Mat> TorchMat::randn(const Shape &shape)
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    torch::Tensor rand_tensor = torch::randn(sizes, torch::kFloat32);
    return std::make_unique<TorchMat>(std::move(rand_tensor));
}

}  // namespace origin
