#include "../../include/mat/array_fire_mat.h"
#include <arrayfire.h>
#include <stdexcept>
#include "../../include/base/utils.h"

namespace origin
{

ArrayFireMat::ArrayFireMat(const std::vector<data_t> &data, const Shape &shape)
{
    // 验证数据是否为空
    if (data.empty())
    {
        throw std::invalid_argument("ArrayFireMat: Tensor data cannot be empty. Data vector is empty.");
    }

    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("ArrayFireMat: Tensor shape cannot have zero dimensions. Dimension " +
                                        std::to_string(i) + " is zero in shape " + shape.to_string());
        }
    }

    af::dim4 dims = convert_shape_to_af_dim4(shape);
    data_         = af::array(dims, data.data());
}

ArrayFireMat::ArrayFireMat(data_t value, const Shape &shape)
{
    // 验证形状是否有效（不能有0维度）
    for (size_t i = 0; i < shape.size(); ++i)
    {
        if (shape[i] == 0)
        {
            throw std::invalid_argument("ArrayFireMat: Tensor shape cannot have zero dimensions. Dimension " +
                                        std::to_string(i) + " is zero in shape " + shape.to_string());
        }
    }

    af::dim4 dims = convert_shape_to_af_dim4(shape);
    data_         = af::constant(value, dims);
}

std::unique_ptr<Mat> ArrayFireMat::clone() const
{
    return std::make_unique<ArrayFireMat>(data_);
}

std::unique_ptr<Mat> ArrayFireMat::reshape(const Shape &shape) const
{
    af::dim4 dims = convert_shape_to_af_dim4(shape);
    return std::make_unique<ArrayFireMat>(af::moddims(data_, dims));
}

std::unique_ptr<Mat> ArrayFireMat::transpose() const
{
    return std::make_unique<ArrayFireMat>(af::transpose(data_));
}

std::unique_ptr<Mat> ArrayFireMat::operator+(const Mat &other) const
{
    const ArrayFireMat &other_af = dynamic_cast<const ArrayFireMat &>(other);
    return std::make_unique<ArrayFireMat>(data_ + other_af.data_);
}

std::unique_ptr<Mat> ArrayFireMat::operator-(const Mat &other) const
{
    const ArrayFireMat &other_af = dynamic_cast<const ArrayFireMat &>(other);
    return std::make_unique<ArrayFireMat>(data_ - other_af.data_);
}

std::unique_ptr<Mat> ArrayFireMat::operator*(const Mat &other) const
{
    const ArrayFireMat &other_af = dynamic_cast<const ArrayFireMat &>(other);
    return std::make_unique<ArrayFireMat>(data_ * other_af.data_);
}

std::unique_ptr<Mat> ArrayFireMat::matmul(const Mat &other) const
{
    const ArrayFireMat &other_af = dynamic_cast<const ArrayFireMat &>(other);
    return std::make_unique<ArrayFireMat>(af::matmul(data_, other_af.data_));
}

std::unique_ptr<Mat> ArrayFireMat::operator/(const Mat &other) const
{
    const ArrayFireMat &other_af = dynamic_cast<const ArrayFireMat &>(other);
    return std::make_unique<ArrayFireMat>(data_ / other_af.data_);
}

std::unique_ptr<Mat> ArrayFireMat::add_scalar(data_t scalar) const
{
    return std::make_unique<ArrayFireMat>(data_ + scalar);
}

std::unique_ptr<Mat> ArrayFireMat::mul_scalar(data_t scalar) const
{
    return std::make_unique<ArrayFireMat>(data_ * scalar);
}

std::unique_ptr<Mat> ArrayFireMat::operator+(data_t scalar) const
{
    return std::make_unique<ArrayFireMat>(data_ + scalar);
}

std::unique_ptr<Mat> ArrayFireMat::operator-(data_t scalar) const
{
    return std::make_unique<ArrayFireMat>(data_ - scalar);
}

std::unique_ptr<Mat> ArrayFireMat::operator*(data_t scalar) const
{
    return std::make_unique<ArrayFireMat>(data_ * scalar);
}

std::unique_ptr<Mat> ArrayFireMat::operator/(data_t scalar) const
{
    return std::make_unique<ArrayFireMat>(data_ / scalar);
}

std::unique_ptr<Mat> ArrayFireMat::operator-() const
{
    return std::make_unique<ArrayFireMat>(-data_);
}

std::unique_ptr<Mat> ArrayFireMat::broadcast_to(const Shape &shape) const
{
    af::dim4 target_dims = convert_shape_to_af_dim4(shape);
    // 使用已有的工具函数，逻辑更完善，性能更优
    auto result = origin::utils::BroadcastTo(data_, target_dims);
    return std::make_unique<ArrayFireMat>(result);
}

std::unique_ptr<Mat> ArrayFireMat::sum_to(const Shape &shape) const
{
    af::dim4 target_dims = convert_shape_to_af_dim4(shape);
    // 使用已有的工具函数，逻辑更完善，性能更优
    auto result = origin::utils::SumTo(data_, target_dims);
    return std::make_unique<ArrayFireMat>(result);
}

std::unique_ptr<Mat> ArrayFireMat::sum(int axis) const
{
    if (axis == -1)
    {
        // 使用af::sum确保返回标量，性能最优
        // 先flatten再sum，确保返回标量
        auto flattened = af::flat(data_);
        auto result    = af::sum(flattened);
        return std::make_unique<ArrayFireMat>(result);
    }
    else
    {
        // ===== ArrayFire与PyTorch行为差异说明 =====
        // ArrayFire使用列主序（column-major）内存布局，PyTorch使用行主序（row-major）
        // 这会导致：
        // 1. 维度压缩：PyTorch会自动压缩求和轴，ArrayFire不会
        // 2. 数据排列：由于内存布局差异，求和结果的数据排列可能不同
        // 3. 形状处理：ArrayFire保持4维结构，PyTorch会压缩到实际维度
        // 注意：此实现可能不匹配PyTorch的预期行为
        // ===========================================
        return std::make_unique<ArrayFireMat>(af::sum(data_, axis));
    }
}

Shape ArrayFireMat::shape() const
{
    return convert_af_dim4_to_shape(data_.dims());
}

size_t ArrayFireMat::elements() const
{
    return data_.elements();
}

std::vector<data_t> ArrayFireMat::to_vector() const
{
    return mat_to_vector(data_);
}

// 数学函数实现
std::unique_ptr<Mat> ArrayFireMat::exp() const
{
    return std::make_unique<ArrayFireMat>(af::exp(data_));
}

std::unique_ptr<Mat> ArrayFireMat::log() const
{
    return std::make_unique<ArrayFireMat>(af::log(data_));
}

std::unique_ptr<Mat> ArrayFireMat::sin() const
{
    return std::make_unique<ArrayFireMat>(af::sin(data_));
}

std::unique_ptr<Mat> ArrayFireMat::cos() const
{
    return std::make_unique<ArrayFireMat>(af::cos(data_));
}

std::unique_ptr<Mat> ArrayFireMat::sqrt() const
{
    return std::make_unique<ArrayFireMat>(af::sqrt(data_));
}

std::unique_ptr<Mat> ArrayFireMat::square() const
{
    return std::make_unique<ArrayFireMat>(data_ * data_);
}

std::unique_ptr<Mat> ArrayFireMat::pow(data_t exponent) const
{
    return std::make_unique<ArrayFireMat>(af::pow(data_, exponent));
}

// 数据访问方法

template <typename T>
T ArrayFireMat::scalar() const
{
    return data_.scalar<T>();
}

// 调试方法
void ArrayFireMat::print(const std::string &desc) const
{
    std::cout << "DL Mat Shape: " << shape() << std::endl;
    if (!desc.empty())
    {
        af::print(desc.c_str(), data_);
    }
    else
    {
        af::print("", data_);
    }
}

// 显式实例化
template data_t ArrayFireMat::scalar<data_t>() const;
template int ArrayFireMat::scalar<int>() const;

data_t ArrayFireMat::sum() const
{
    return af::sum<data_t>(data_);
}

data_t ArrayFireMat::max() const
{
    return af::max<data_t>(data_);
}

data_t ArrayFireMat::min() const
{
    return af::min<data_t>(data_);
}

data_t ArrayFireMat::mean() const
{
    return af::mean<data_t>(data_);
}

// 静态辅助函数实现
std::vector<data_t> ArrayFireMat::mat_to_vector(const af::array &arr)
{
    std::vector<data_t> result(arr.elements());
    arr.host(result.data());
    return result;
}

af::array ArrayFireMat::vector_to_mat(const std::vector<data_t> &data, const Shape &shape)
{
    af::dim4 dims = convert_shape_to_af_dim4(shape);
    return af::array(dims, data.data());
}

af::dim4 ArrayFireMat::convert_shape_to_af_dim4(const Shape &shape)
{
    const auto &dims = shape.dims();
    if (dims.size() == 0)
    {
        return af::dim4(1);
    }
    else if (dims.size() == 1)
    {
        return af::dim4(dims[0]);
    }
    else if (dims.size() == 2)
    {
        return af::dim4(dims[0], dims[1]);
    }
    else if (dims.size() == 3)
    {
        return af::dim4(dims[0], dims[1], dims[2]);
    }
    else if (dims.size() == 4)
    {
        return af::dim4(dims[0], dims[1], dims[2], dims[3]);
    }
    else
    {
        // 对于超过4维的情况，填充到4维
        logw("shape is too large, fill to 4 dims");
        af::dim4 result(1, 1, 1, 1);
        for (size_t i = 0; i < std::min(dims.size(), size_t(4)); ++i)
        {
            result[i] = dims[i];
        }
        return result;
    }
}

Shape ArrayFireMat::convert_af_dim4_to_shape(const af::dim4 &dims)
{
    std::vector<size_t> shape_dims;
    for (int i = 0; i < 4; ++i)
    {
        if (dims[i] > 1 || i == 0)
        {  // 保留至少一个维度
            shape_dims.push_back(dims[i]);
        }
    }
    return Shape(shape_dims);
}

// 静态工厂方法实现
std::unique_ptr<Mat> ArrayFireMat::randn(const Shape &shape)
{
    af::dim4 dims      = convert_shape_to_af_dim4(shape);
    af::array rand_mat = af::randn(dims);
    return std::make_unique<ArrayFireMat>(std::move(rand_mat));
}

}  // namespace origin
