#ifndef __ORIGIN_DL_TORCH_MAT_H__
#define __ORIGIN_DL_TORCH_MAT_H__

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "../../core/tensor_options.h"
#include "../basic_types.h"
#include "../mat.h"
#include "../shape.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"

namespace origin
{

/**
 * @brief Torch后端的矩阵实现
 * @details 继承自Mat接口，使用LibTorch库实现矩阵计算
 */
class TorchMat : public Mat
{
public:
    // 将data_设为public，便于友元类访问
    torch::Tensor data_;

public:
    /**
     * @brief 默认构造函数
     */
    TorchMat() = default;

    /**
     * @brief 从Torch张量构造
     * @param tensor Torch张量
     */
    explicit TorchMat(const torch::Tensor &tensor) : data_(tensor) {}

    /**
     * @brief 从Torch张量移动构造
     * @param tensor Torch张量
     */
    explicit TorchMat(torch::Tensor &&tensor) : data_(std::move(tensor)) {}

    /**
     * @brief 通用构造函数：从不同数据类型创建
     * @param data 数据向量
     * @param shape 矩阵形状
     */
    template <typename T>
    TorchMat(const std::vector<T> &data, const Shape &shape)
    {
        // 验证数据是否为空
        if (data.empty())
        {
            throw std::invalid_argument("TorchMat: Tensor data cannot be empty. Data vector is empty.");
        }

        // 验证形状是否有效
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
            {
                throw std::invalid_argument("TorchMat: Tensor shape cannot have zero dimensions. Dimension " +
                                            std::to_string(i) + " is zero in shape " + shape.to_string());
            }
        }

        auto sizes      = TorchMat::convert_shape_to_torch_sizes(shape);
        auto data_type  = DataTypeTraits<T>::type;
        auto torch_type = get_torch_type(data_type);
        data_           = torch::from_blob(const_cast<T *>(data.data()), sizes, torch_type).clone();
    }

    /**
     * @brief 通用构造函数：从标量创建
     * @param value 标量值
     * @param shape 矩阵形状
     */
    template <typename T>
    TorchMat(T value, const Shape &shape)
    {
        // 验证形状是否有效
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
            {
                throw std::invalid_argument("TorchMat: Tensor shape cannot have zero dimensions. Dimension " +
                                            std::to_string(i) + " is zero in shape " + shape.to_string());
            }
        }

        auto sizes      = TorchMat::convert_shape_to_torch_sizes(shape);
        auto data_type  = DataTypeTraits<T>::type;
        auto torch_type = get_torch_type(data_type);
        data_           = torch::full(sizes, static_cast<T>(value), torch_type);
    }

    /**
     * @brief 通用构造函数：从数据创建（支持TensorOptions）
     * @param data 数据向量
     * @param shape 矩阵形状
     * @param options 张量选项
     */
    template <typename T>
    TorchMat(const std::vector<T> &data, const Shape &shape, const TensorOptions &options)
    {
        // 验证数据是否为空
        if (data.empty())
        {
            throw std::invalid_argument("TorchMat: Tensor data cannot be empty. Data vector is empty.");
        }

        // 验证形状是否有效
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
            {
                throw std::invalid_argument("TorchMat: Tensor shape cannot have zero dimensions. Dimension " +
                                            std::to_string(i) + " is zero in shape " + shape.to_string());
            }
        }

        auto sizes         = TorchMat::convert_shape_to_torch_sizes(shape);
        auto torch_options = get_torch_tensor_options(options);
        data_              = torch::from_blob(const_cast<T *>(data.data()), sizes, torch_options).clone();
    }

    /**
     * @brief 通用构造函数：从标量创建（支持TensorOptions）
     * @param value 标量值
     * @param shape 矩阵形状
     * @param options 张量选项
     */
    template <typename T>
    TorchMat(T value, const Shape &shape, const TensorOptions &options)
    {
        // 验证形状是否有效
        for (size_t i = 0; i < shape.size(); ++i)
        {
            if (shape[i] == 0)
            {
                throw std::invalid_argument("TorchMat: Tensor shape cannot have zero dimensions. Dimension " +
                                            std::to_string(i) + " is zero in shape " + shape.to_string());
            }
        }

        auto sizes         = TorchMat::convert_shape_to_torch_sizes(shape);
        auto torch_options = get_torch_tensor_options(options);
        data_              = torch::full(sizes, static_cast<T>(value), torch_options);
    }

    // 实现Mat接口的所有虚函数
    std::unique_ptr<Mat> clone() const override;
    std::unique_ptr<Mat> view(const Shape &shape) const override;
    bool is_contiguous() const override;
    std::unique_ptr<Mat> contiguous() const override;
    std::unique_ptr<Mat> reshape(const Shape &shape) const override;
    std::unique_ptr<Mat> transpose() const override;

    // 兼容性方法
    std::unique_ptr<Mat> T() const { return transpose(); }
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    void add_inplace(const Mat &other) override;
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    void sub_inplace(const Mat &other) override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    void mul_inplace(const Mat &other) override;
    std::unique_ptr<Mat> matmul(const Mat &other) const override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    void div_inplace(const Mat &other) override;

    std::unique_ptr<Mat> operator-() const override;
    std::unique_ptr<Mat> broadcast_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum(int axis = -1) const override;
    Shape shape() const override;
    size_t elements() const override;
    // 虚函数重写
    std::vector<float> to_vector() const override;

    // 模板版本
    template <typename U>
    std::vector<U> to_vector() const;

    // 数学函数
    std::unique_ptr<Mat> exp() const override;
    void exp_inplace() override;
    std::unique_ptr<Mat> log() const override;
    void log_inplace() override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;
    void sqrt_inplace() override;
    std::unique_ptr<Mat> square() const override;
    void square_inplace() override;
    // 虚函数重写（与OriginMat接口对齐）
    std::unique_ptr<Mat> pow(const Scalar &exponent) const override;
    void pow_inplace(const Scalar &exponent) override;
    std::unique_ptr<Mat> relu() const override;
    void relu_inplace() override;
    void neg_inplace() override;

    // 0维张量支持（与OriginMat接口对齐）
    bool is_scalar() const override;
    Scalar scalar_value() const override;

    // 模板版本
    template <typename U>
    std::unique_ptr<Mat> pow(U exponent) const;

    // 数据访问
    template <typename U>
    U scalar() const;
    int backend_type() const override;

    // 新增：类型相关方法
    DataType dtype() const override;
    std::unique_ptr<Mat> to(DataType target_type) const override;

    // 新增：设备相关方法
    Device device() const override;
    std::unique_ptr<Mat> to_device(Device device) const override;

    // === 泛型数据访问方法 ===
    void *data_ptr() override;

    template <typename U>
    U *data_ptr();

    // 调试方法
    void print(const std::string &desc = "") const override;

    /**
     * @brief 静态辅助函数：将Torch张量转换为向量
     * @param tensor Torch张量
     * @return 数据向量
     */
    template <typename U>
    static std::vector<U> tensor_to_vector(const torch::Tensor &tensor);

    /**
     * @brief 静态辅助函数：将向量转换为Torch张量
     * @param data 数据向量
     * @param shape 矩阵形状
     * @return Torch张量
     */
    template <typename U>
    static torch::Tensor vector_to_tensor(const std::vector<U> &data, const Shape &shape);

    /**
     * @brief 静态辅助函数：将Shape转换为torch::IntArrayRef
     * @param shape Shape对象
     * @return std::vector<int64_t>对象
     */
    static std::vector<int64_t> convert_shape_to_torch_sizes(const Shape &shape);

    /**
     * @brief 静态辅助函数：将torch::IntArrayRef转换为Shape
     * @param sizes torch::IntArrayRef对象
     * @return Shape对象
     */
    static Shape convert_torch_sizes_to_shape(const torch::IntArrayRef &sizes);

    /**
     * @brief 静态工厂方法：创建随机数矩阵
     * @param shape 矩阵形状
     * @return 随机数矩阵
     */
    static std::unique_ptr<Mat> randn(const Shape &shape);

    /**
     * @brief 静态工厂方法：创建随机数矩阵（支持TensorOptions）
     * @param shape 矩阵形状
     * @param options 张量选项
     * @return 随机数矩阵
     */
    static std::unique_ptr<Mat> randn(const Shape &shape, const TensorOptions &options);

    static std::unique_ptr<Mat> from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options);

    static std::unique_ptr<Mat> from_memory(const void *data,
                                            DataType user_dtype,
                                            const Shape &shape,
                                            const TensorOptions &options);

private:
    // 辅助：根据 DataType 和 Scalar 生成 torch::Scalar（通过类型分发器）
    static torch::Scalar make_torch_scalar_from_scalar(const Scalar &scalar, DataType dtype);

    /**
     * @brief 将DataType转换为torch::ScalarType
     * @param dtype DataType枚举
     * @return 对应的torch::ScalarType
     */
    static torch::ScalarType get_torch_type(DataType dtype);

    /**
     * @brief 将torch::ScalarType转换为DataType
     * @param torch_type torch::ScalarType
     * @return 对应的DataType
     */
    static DataType get_data_type_from_torch(torch::ScalarType torch_type);

    /**
     * @brief 将TensorOptions转换为torch::TensorOptions
     * @param options 张量选项
     * @return 对应的torch::TensorOptions
     */
    static torch::TensorOptions get_torch_tensor_options(const TensorOptions &options);
};

}  // namespace origin

#endif  // __ORIGIN_DL_TORCH_MAT_H__
