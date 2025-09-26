#ifndef __ORIGIN_DL_ARRAYFIRE_MAT_H__
#define __ORIGIN_DL_ARRAYFIRE_MAT_H__

#include <arrayfire.h>
#include <memory>
#include <vector>
#include "dlMat.h"
#include "dlShape.h"

namespace dl
{

/**
 * @brief ArrayFire后端的矩阵实现
 * @details 继承自Mat接口，使用ArrayFire库实现矩阵计算
 */
class ArrayFireMat : public Mat
{
public:
    // 将data_设为public，便于友元类访问
    af::array data_;

public:
    /**
     * @brief 默认构造函数
     */
    ArrayFireMat() = default;

    /**
     * @brief 从ArrayFire数组构造
     * @param arr ArrayFire数组
     */
    explicit ArrayFireMat(const af::array &arr) : data_(arr) {}

    /**
     * @brief 从ArrayFire数组移动构造
     * @param arr ArrayFire数组
     */
    explicit ArrayFireMat(af::array &&arr) : data_(std::move(arr)) {}

    /**
     * @brief 从数据向量构造
     * @param data 数据向量
     * @param shape 矩阵形状
     */
    ArrayFireMat(const std::vector<double> &data, const Shape &shape);

    /**
     * @brief 从数据指针构造
     * @param data 数据指针
     * @param shape 矩阵形状
     */
    ArrayFireMat(const double *data, const Shape &shape);

    /**
     * @brief 从标量值构造常量矩阵
     * @param value 标量值
     * @param shape 矩阵形状
     */
    ArrayFireMat(data_t value, const Shape &shape);

    // 实现Mat接口的所有虚函数
    std::unique_ptr<Mat> clone() const override;
    std::unique_ptr<Mat> reshape(const Shape &shape) const override;
    std::unique_ptr<Mat> transpose() const override;

    // 兼容性方法
    std::unique_ptr<Mat> T() const { return transpose(); }
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    std::unique_ptr<Mat> add_scalar(double scalar) const override;
    std::unique_ptr<Mat> mul_scalar(double scalar) const override;
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
    std::vector<double> to_vector() const override;

    // 数学函数
    std::unique_ptr<Mat> exp() const override;
    std::unique_ptr<Mat> log() const override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;
    std::unique_ptr<Mat> square() const override;
    std::unique_ptr<Mat> pow(double exponent) const override;

    // 数据访问
    template <typename T>
    T scalar() const;
    double sum() const override;
    double max() const override;
    double min() const override;
    double mean() const override;

    // 调试方法
    void print(const std::string &desc = "") const override;

    /**
     * @brief 静态辅助函数：将ArrayFire数组转换为向量
     * @param arr ArrayFire数组
     * @return 数据向量
     */
    static std::vector<double> array_to_vector(const af::array &arr);

    /**
     * @brief 静态辅助函数：将向量转换为ArrayFire数组
     * @param data 数据向量
     * @param shape 矩阵形状
     * @return ArrayFire数组
     */
    static af::array vector_to_array(const std::vector<double> &data, const Shape &shape);

    /**
     * @brief 静态辅助函数：将Shape转换为af::dim4
     * @param shape Shape对象
     * @return af::dim4对象
     */
    static af::dim4 convert_shape_to_af_dim4(const Shape &shape);

    /**
     * @brief 静态辅助函数：将af::dim4转换为Shape
     * @param dims af::dim4对象
     * @return Shape对象
     */
    static Shape convert_af_dim4_to_shape(const af::dim4 &dims);
};

}  // namespace dl

#endif  // __ORIGIN_DL_ARRAYFIRE_MAT_H__
