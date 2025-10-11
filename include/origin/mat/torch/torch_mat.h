#ifndef __ORIGIN_DL_TORCH_MAT_H__
#define __ORIGIN_DL_TORCH_MAT_H__

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "../basic_types.h"
#include "../mat.h"
#include "../shape.h"

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
        auto data_type  = get_data_type_from_template<T>();
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
        auto data_type  = get_data_type_from_template<T>();
        auto torch_type = get_torch_type(data_type);
        data_           = torch::full(sizes, static_cast<T>(value), torch_type);
    }

    // 实现Mat接口的所有虚函数
    std::unique_ptr<Mat> clone() const override;
    std::unique_ptr<Mat> reshape(const Shape &shape) const override;
    std::unique_ptr<Mat> transpose() const override;

    // 兼容性方法
    std::unique_ptr<Mat> T() const { return transpose(); }
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    std::unique_ptr<Mat> matmul(const Mat &other) const override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    // 虚函数重写（保持与基类兼容）
    std::unique_ptr<Mat> add_scalar(data_t scalar) const override;
    std::unique_ptr<Mat> mul_scalar(data_t scalar) const override;
    std::unique_ptr<Mat> operator+(data_t scalar) const override;
    std::unique_ptr<Mat> operator-(data_t scalar) const override;
    std::unique_ptr<Mat> operator*(data_t scalar) const override;
    std::unique_ptr<Mat> operator/(data_t scalar) const override;
    
    // 模板版本（提供泛型支持）
    template <typename U>
    std::unique_ptr<Mat> add_scalar(U scalar) const;
    template <typename U>
    std::unique_ptr<Mat> mul_scalar(U scalar) const;
    template <typename U>
    std::unique_ptr<Mat> operator+(U scalar) const;
    template <typename U>
    std::unique_ptr<Mat> operator-(U scalar) const;
    template <typename U>
    std::unique_ptr<Mat> operator*(U scalar) const;
    template <typename U>
    std::unique_ptr<Mat> operator/(U scalar) const;
    std::unique_ptr<Mat> operator-() const override;
    std::unique_ptr<Mat> broadcast_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum(int axis = -1) const override;
    Shape shape() const override;
    size_t elements() const override;
    // 虚函数重写
    std::vector<data_t> to_vector() const override;
    
    // 模板版本
    template <typename U>
    std::vector<U> to_vector() const;

    // 数学函数
    std::unique_ptr<Mat> exp() const override;
    std::unique_ptr<Mat> log() const override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;
    std::unique_ptr<Mat> square() const override;
    // 虚函数重写
    std::unique_ptr<Mat> pow(data_t exponent) const override;
    
    // 模板版本
    template <typename U>
    std::unique_ptr<Mat> pow(U exponent) const;

    // 数据访问
    template <typename U>
    U scalar() const;
    data_t sum_all() const override;
    data_t max_all() const override;
    data_t min_all() const override;
    data_t mean_all() const override;
    int backend_type() const override;

    // 新增：类型相关方法
    DataType dtype() const override;
    std::unique_ptr<Mat> to(DataType target_type) const override;

    // === 泛型数据访问方法 ===
    template <typename U>
    U* data_ptr();

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

private:
    /**
     * @brief 类型推断辅助函数
     * @return 对应的DataType
     */
    template <typename T>
    DataType get_data_type() const
    {
        return get_data_type_from_template<T>();
    }

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
};

}  // namespace origin

#endif  // __ORIGIN_DL_TORCH_MAT_H__
