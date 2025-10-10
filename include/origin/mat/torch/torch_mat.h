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
     * @brief 从数据向量构造
     * @param data 数据向量
     * @param shape 矩阵形状
     */
    TorchMat(const std::vector<data_t> &data, const Shape &shape);

    /**
     * @brief 从标量值构造常量矩阵
     * @param value 标量值
     * @param shape 矩阵形状
     */
    TorchMat(data_t value, const Shape &shape);

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
    std::unique_ptr<Mat> add_scalar(data_t scalar) const override;
    std::unique_ptr<Mat> mul_scalar(data_t scalar) const override;
    std::unique_ptr<Mat> operator+(data_t scalar) const override;
    std::unique_ptr<Mat> operator-(data_t scalar) const override;
    std::unique_ptr<Mat> operator*(data_t scalar) const override;
    std::unique_ptr<Mat> operator/(data_t scalar) const override;
    std::unique_ptr<Mat> operator-() const override;
    std::unique_ptr<Mat> broadcast_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum(int axis = -1) const override;
    Shape shape() const override;
    size_t elements() const override;
    std::vector<data_t> to_vector() const override;

    // 数学函数
    std::unique_ptr<Mat> exp() const override;
    std::unique_ptr<Mat> log() const override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;
    std::unique_ptr<Mat> square() const override;
    std::unique_ptr<Mat> pow(data_t exponent) const override;

    // 数据访问
    template <typename T>
    T scalar() const;
    data_t sum() const override;
    data_t max() const override;
    data_t min() const override;
    data_t mean() const override;
    int backend_type() const override;

    // 调试方法
    void print(const std::string &desc = "") const override;

    /**
     * @brief 静态辅助函数：将Torch张量转换为向量
     * @param tensor Torch张量
     * @return 数据向量
     */
    static std::vector<data_t> tensor_to_vector(const torch::Tensor &tensor);

    /**
     * @brief 静态辅助函数：将向量转换为Torch张量
     * @param data 数据向量
     * @param shape 矩阵形状
     * @return Torch张量
     */
    static torch::Tensor vector_to_tensor(const std::vector<data_t> &data, const Shape &shape);

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
};

}  // namespace origin

#endif  // __ORIGIN_DL_TORCH_MAT_H__
