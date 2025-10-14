#include "origin/mat/torch/torch_mat.h"
#include <torch/torch.h>
#include <stdexcept>
#include "origin/core/tensor_options.h"
#include "origin/mat/basic_types.h"
#include "origin/utils/log.h"

namespace origin
{

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
    /*
    对于2D张量：交换行和列（完全转置）
    对于高维张量：只交换最后两个维度
    # 对于2D张量
    x = torch.randn(3, 4)
    x_t = x.T  # 结果形状为 (4, 3)

    # 对于高维张量
    y = torch.randn(2, 3, 4, 5)
    y_t = y.T  # 结果形状为 (2, 3, 5, 4)
    */
    auto dims = data_.dim();
    if (dims < 2)
    {
        // 一维张量转置返回自身
        return std::make_unique<TorchMat>(data_);
    }
    else
    {
        // 二维及以上张量转置最后两个维度
        return std::make_unique<TorchMat>(data_.transpose(-2, -1));
    }
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

// 虚函数重写实现 - 调用模板版本
std::unique_ptr<Mat> TorchMat::add_scalar(data_t scalar) const
{
    return add_scalar<data_t>(scalar);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::add_scalar(U scalar) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor scalar_tensor = torch::full({}, scalar, torch_type);
    return std::make_unique<TorchMat>(data_ + scalar_tensor);
}

std::unique_ptr<Mat> TorchMat::mul_scalar(data_t scalar) const
{
    return mul_scalar<data_t>(scalar);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::mul_scalar(U scalar) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor scalar_tensor = torch::full({}, scalar, torch_type);
    return std::make_unique<TorchMat>(data_ * scalar_tensor);
}

std::unique_ptr<Mat> TorchMat::operator+(data_t scalar) const
{
    return operator+ <data_t>(scalar);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::operator+(U scalar) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor scalar_tensor = torch::full({}, scalar, torch_type);
    return std::make_unique<TorchMat>(data_ + scalar_tensor);
}

std::unique_ptr<Mat> TorchMat::operator-(data_t scalar) const
{
    return operator- <data_t>(scalar);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::operator-(U scalar) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor scalar_tensor = torch::full({}, scalar, torch_type);
    return std::make_unique<TorchMat>(data_ - scalar_tensor);
}

std::unique_ptr<Mat> TorchMat::operator*(data_t scalar) const
{
    return operator* <data_t>(scalar);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::operator*(U scalar) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor scalar_tensor = torch::full({}, scalar, torch_type);
    return std::make_unique<TorchMat>(data_ * scalar_tensor);
}

std::unique_ptr<Mat> TorchMat::operator/(data_t scalar) const
{
    return operator/ <data_t>(scalar);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::operator/(U scalar) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor scalar_tensor = torch::full({}, scalar, torch_type);
    return std::make_unique<TorchMat>(data_ / scalar_tensor);
}

std::unique_ptr<Mat> TorchMat::operator-() const
{
    return std::make_unique<TorchMat>(-data_);
}

std::unique_ptr<Mat> TorchMat::broadcast_to(const Shape &shape) const
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);
    // 使用clone()确保返回实际的数据副本，而不是视图
    // 注意：LibTorch的expand()方法遵循严格的广播规则：
    // 1. 从右到左比较维度大小
    // 2. 每个维度要么大小相同，要么其中一个为1，要么其中一个不存在
    // 3. 如果违反规则会抛出"expanded size must match existing size"异常
    return std::make_unique<TorchMat>(data_.expand(sizes).clone());
}

std::unique_ptr<Mat> TorchMat::sum_to(const Shape &shape) const
{
    auto sizes  = TorchMat::convert_shape_to_torch_sizes(shape);
    auto result = data_;

    // 计算需要求和的维度
    auto current_sizes = data_.sizes();
    auto target_sizes  = sizes;

    // 如果源数组已经是目标形状，则直接返回
    if (current_sizes == target_sizes)
    {
        return std::make_unique<TorchMat>(result);
    }

    // 计算元素总数
    size_t current_elements = 1;
    for (auto dim : current_sizes)
    {
        current_elements *= dim;
    }

    size_t target_elements = 1;
    for (auto dim : target_sizes)
    {
        target_elements *= dim;
    }

    if (target_elements > current_elements)
    {
        // 目标形状更大，libtorch的sum_to不支持广播，抛出异常
        throw std::runtime_error("sum_to: Target shape cannot have more elements than source tensor");
    }
    else
    {
        // 目标形状更小或相等，需要求和压缩
        // 收集需要求和的维度
        std::vector<int> sum_dims;
        for (size_t i = 0; i < std::min(current_sizes.size(), target_sizes.size()); ++i)
        {
            if (target_sizes[i] == 1 && current_sizes[i] > 1)
            {
                sum_dims.push_back(i);
            }
        }

        // 处理多余的维度
        for (size_t i = target_sizes.size(); i < current_sizes.size(); ++i)
        {
            sum_dims.push_back(i);
        }

        // 一次性对所有需要求和的维度进行求和
        // 注意：需要从大到小排序，避免维度索引变化
        std::sort(sum_dims.begin(), sum_dims.end(), std::greater<int>());
        for (int dim : sum_dims)
        {
            result = result.sum(dim, false);
        }

        // 确保结果的形状正确
        if (result.sizes() != target_sizes)
        {
            result = result.reshape(sizes);
        }
    }

    return std::make_unique<TorchMat>(result);
}

std::unique_ptr<Mat> TorchMat::sum(int axis) const
{
    if (axis == -1)
    {
        // 对所有元素求和，返回形状为[1]的张量（与PyTorch测试期望一致）
        auto result = data_.sum();
        // 将标量转换为形状为[1]的张量
        return std::make_unique<TorchMat>(result.unsqueeze(0));
    }
    else
    {
        // 沿指定轴求和，压缩维度（与PyTorch默认行为一致）
        return std::make_unique<TorchMat>(data_.sum(axis, false));
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
    // 根据张量的实际类型进行转换
    if (data_.scalar_type() == torch::kFloat32)
    {
        return to_vector<float>();
    }
    else if (data_.scalar_type() == torch::kFloat64)
    {
        auto double_vec = to_vector<double>();
        std::vector<data_t> result;
        result.reserve(double_vec.size());
        for (const auto &val : double_vec)
        {
            result.push_back(static_cast<data_t>(val));
        }
        return result;
    }
    else if (data_.scalar_type() == torch::kInt32)
    {
        auto int_vec = to_vector<int32_t>();
        std::vector<data_t> result;
        result.reserve(int_vec.size());
        for (const auto &val : int_vec)
        {
            result.push_back(static_cast<data_t>(val));
        }
        return result;
    }
    else if (data_.scalar_type() == torch::kInt8)
    {
        auto int_vec = to_vector<int8_t>();
        std::vector<data_t> result;
        result.reserve(int_vec.size());
        for (const auto &val : int_vec)
        {
            result.push_back(static_cast<data_t>(val));
        }
        return result;
    }
    else
    {
        throw std::runtime_error("Unsupported tensor type for to_vector()");
    }
}

template <typename U>
std::vector<U> TorchMat::to_vector() const
{
    return TorchMat::tensor_to_vector<U>(data_);
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
    return pow<data_t>(exponent);
}

template <typename U>
std::unique_ptr<Mat> TorchMat::pow(U exponent) const
{
    // 根据输入类型动态转换
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);

    // 创建标量tensor
    torch::Tensor exponent_tensor = torch::full({}, exponent, torch_type);
    return std::make_unique<TorchMat>(torch::pow(data_, exponent_tensor));
}

// 数据访问方法
template <typename U>
U TorchMat::scalar() const
{
    return data_.item<U>();
}

// 调试方法
void TorchMat::print(const std::string &desc) const
{
    if (!desc.empty())
    {
        std::cout << desc << ": " << std::endl;
    }
    std::cout << data_ << std::endl;
}

// 显式实例化
template data_t TorchMat::scalar<data_t>() const;
template int TorchMat::scalar<int>() const;

// 全局聚合函数实现 - 返回标量值
// 注意：这些函数重命名为 *_all() 是为了避免与 sum(int axis) 等按轴操作的函数名冲突
data_t TorchMat::sum_all() const
{
    return data_.sum().item<data_t>();
}

data_t TorchMat::max_all() const
{
    return data_.max().item<data_t>();
}

data_t TorchMat::min_all() const
{
    return data_.min().item<data_t>();
}

data_t TorchMat::mean_all() const
{
    return data_.mean().item<data_t>();
}

int TorchMat::backend_type() const
{
    return TORCH_BACKEND_TYPE;
}

/*
视图转置 vs 数据转置：PyTorch的transpose()是视图操作，不重新排列内存数据
连续性检查：使用contiguous()确保数据按逻辑形状排列
性能考虑：contiguous()只在需要时重新排列数据，避免不必要的内存拷贝
*/
// 静态辅助函数实现
template <typename U>
std::vector<U> TorchMat::tensor_to_vector(const torch::Tensor &tensor)
{
    std::vector<U> result(tensor.numel());
    // 确保张量是连续的，这样数据会按照逻辑形状重新排列。
    // 考虑到转置的情况，使用视图转置，数据的内存顺序不会改变。所以直接返回tensor.data_ptr<U>()导致看不出转置的效果。
    auto contiguous_tensor = tensor.contiguous();
    auto data_ptr = contiguous_tensor.data_ptr<U>();
    std::copy(data_ptr, data_ptr + tensor.numel(), result.begin());
    return result;
}

template <typename U>
torch::Tensor TorchMat::vector_to_tensor(const std::vector<U> &data, const Shape &shape)
{
    auto sizes      = TorchMat::convert_shape_to_torch_sizes(shape);
    auto data_type  = get_data_type_from_template<U>();
    auto torch_type = get_torch_type(data_type);
    return torch::from_blob(const_cast<U *>(data.data()), sizes, torch_type).clone();
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
    auto sizes                = TorchMat::convert_shape_to_torch_sizes(shape);
    torch::Tensor rand_tensor = torch::randn(sizes, torch::kFloat32);
    return std::make_unique<TorchMat>(std::move(rand_tensor));
}

std::unique_ptr<Mat> TorchMat::randn(const Shape &shape, const TensorOptions &options)
{
    auto sizes = TorchMat::convert_shape_to_torch_sizes(shape);

    // 对于非浮点类型，先生成float32再转换
    if (options.dtype() == DataType::kFloat32 || options.dtype() == DataType::kDouble)
    {
        auto torch_options        = get_torch_tensor_options(options);
        torch::Tensor rand_tensor = torch::randn(sizes, torch_options);
        return std::make_unique<TorchMat>(std::move(rand_tensor));
    }
    else
    {
        // 对于整数类型，先生成float32再转换
        torch::Tensor rand_tensor = torch::randn(sizes, torch::kFloat32);
        auto result               = std::make_unique<TorchMat>(std::move(rand_tensor));

        // 转换到目标类型
        auto converted = result->to(options.dtype());

        // 如果设备不是CPU，移动到指定设备
        if (options.device().type() != DeviceType::kCPU)
        {
            converted = converted->to_device(options.device());
        }

        return converted;
    }
}

// 类型相关方法实现
DataType TorchMat::dtype() const
{
    return get_data_type_from_torch(data_.scalar_type());
}

std::unique_ptr<Mat> TorchMat::to(DataType target_type) const
{
    auto torch_type       = get_torch_type(target_type);
    auto converted_tensor = data_.to(torch_type);
    return std::make_unique<TorchMat>(std::move(converted_tensor));
}

Device TorchMat::device() const
{
    auto torch_device = data_.device();
    if (torch_device.is_cpu())
    {
        return Device(DeviceType::kCPU);
    }
    else if (torch_device.is_cuda())
    {
        return Device(DeviceType::kCUDA, torch_device.index());
    }
    else
    {
        throw std::runtime_error("Unsupported device type");
    }
}

std::unique_ptr<Mat> TorchMat::to_device(Device device) const
{
    torch::Device torch_device = torch::kCPU;  // 默认初始化为CPU
    if (device.type() == DeviceType::kCPU)
    {
        torch_device = torch::kCPU;
    }
    else if (device.type() == DeviceType::kCUDA)
    {
        torch_device = torch::Device(torch::kCUDA, device.index());
    }
    else
    {
        throw std::invalid_argument("Unsupported device type");
    }

    auto moved_tensor = data_.to(torch_device);
    return std::make_unique<TorchMat>(std::move(moved_tensor));
}

torch::ScalarType TorchMat::get_torch_type(DataType dtype)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            return torch::kFloat32;
        case DataType::kDouble:
            return torch::kFloat64;
        case DataType::kInt32:
            return torch::kInt32;
        case DataType::kInt8:
            return torch::kInt8;
        default:
            throw std::invalid_argument("Unsupported data type");
    }
}

DataType TorchMat::get_data_type_from_torch(torch::ScalarType torch_type)
{
    switch (torch_type)
    {
        case torch::kFloat32:
            return DataType::kFloat32;
        case torch::kFloat64:
            return DataType::kDouble;
        case torch::kInt32:
            return DataType::kInt32;
        case torch::kInt8:
            return DataType::kInt8;
        default:
            throw std::invalid_argument("Unsupported torch scalar type");
    }
}

torch::TensorOptions TorchMat::get_torch_tensor_options(const TensorOptions &options)
{
    auto torch_options = torch::TensorOptions().dtype(get_torch_type(options.dtype()));

    if (options.device().type() == DeviceType::kCUDA)
    {
        torch_options = torch_options.device(torch::kCUDA, options.device().index());
    }
    else
    {
        torch_options = torch_options.device(torch::kCPU);
    }

    return torch_options;
}

// === TorchMat泛型方法实现 ===
template <typename U>
U *TorchMat::data_ptr()
{
    return data_.data_ptr<U>();
}

// === 模板实例化 ===
template float *TorchMat::data_ptr<float>();
template double *TorchMat::data_ptr<double>();
template int32_t *TorchMat::data_ptr<int32_t>();
template int8_t *TorchMat::data_ptr<int8_t>();

// 标量操作模板实例化（注释掉，让编译器自动实例化）
// template std::unique_ptr<Mat> TorchMat::add_scalar<float>(float scalar) const;
// template std::unique_ptr<Mat> TorchMat::add_scalar<double>(double scalar) const;
// template std::unique_ptr<Mat> TorchMat::add_scalar<int32_t>(int32_t scalar) const;
// template std::unique_ptr<Mat> TorchMat::add_scalar<int8_t>(int8_t scalar) const;

// template std::unique_ptr<Mat> TorchMat::mul_scalar<float>(float scalar) const;
// template std::unique_ptr<Mat> TorchMat::mul_scalar<double>(double scalar) const;
// template std::unique_ptr<Mat> TorchMat::mul_scalar<int32_t>(int32_t scalar) const;
// template std::unique_ptr<Mat> TorchMat::mul_scalar<int8_t>(int8_t scalar) const;

// template std::unique_ptr<Mat> TorchMat::operator+<float>(float scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator+<double>(double scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator+<int32_t>(int32_t scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator+<int8_t>(int8_t scalar) const;

// template std::unique_ptr<Mat> TorchMat::operator-<float>(float scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator-<double>(double scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator-<int32_t>(int32_t scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator-<int8_t>(int8_t scalar) const;

// template std::unique_ptr<Mat> TorchMat::operator*<float>(float scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator*<double>(double scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator*<int32_t>(int32_t scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator*<int8_t>(int8_t scalar) const;

// template std::unique_ptr<Mat> TorchMat::operator/<float>(float scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator/<double>(double scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator/<int32_t>(int32_t scalar) const;
// template std::unique_ptr<Mat> TorchMat::operator/<int8_t>(int8_t scalar) const;

// pow 模板实例化（如果需要的话）
// template std::unique_ptr<Mat> TorchMat::pow<float>(float exponent) const;
// template std::unique_ptr<Mat> TorchMat::pow<double>(double exponent) const;
// template std::unique_ptr<Mat> TorchMat::pow<int32_t>(int32_t exponent) const;
// template std::unique_ptr<Mat> TorchMat::pow<int8_t>(int8_t exponent) const;

template std::vector<float> TorchMat::to_vector<float>() const;
template std::vector<double> TorchMat::to_vector<double>() const;
template std::vector<int32_t> TorchMat::to_vector<int32_t>() const;
template std::vector<int8_t> TorchMat::to_vector<int8_t>() const;

template std::vector<float> TorchMat::tensor_to_vector<float>(const torch::Tensor &tensor);
template std::vector<double> TorchMat::tensor_to_vector<double>(const torch::Tensor &tensor);
template std::vector<int32_t> TorchMat::tensor_to_vector<int32_t>(const torch::Tensor &tensor);
template std::vector<int8_t> TorchMat::tensor_to_vector<int8_t>(const torch::Tensor &tensor);

// vector_to_tensor 模板实例化（如果需要的话）
// template torch::Tensor TorchMat::vector_to_tensor<float>(const std::vector<float> &data, const Shape &shape);
// template torch::Tensor TorchMat::vector_to_tensor<double>(const std::vector<double> &data, const Shape &shape);
// template torch::Tensor TorchMat::vector_to_tensor<int32_t>(const std::vector<int32_t> &data, const Shape &shape);
// template torch::Tensor TorchMat::vector_to_tensor<int8_t>(const std::vector<int8_t> &data, const Shape &shape);

}  // namespace origin
