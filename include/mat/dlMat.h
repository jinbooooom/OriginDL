#ifndef __ORIGIN_DL_MAT_H__
#define __ORIGIN_DL_MAT_H__

#include <memory>
#include <vector>
#include "dlShape.h"
#include "../base/dlTypes.h"

namespace dl
{

// 前向声明
class Shape;



/**
 * @brief 矩阵计算抽象接口
 * @details 定义了矩阵计算的基本操作，支持多种后端实现
 */
class Mat
{
public:
    virtual ~Mat() = default;

    /**
     * @brief 克隆矩阵
     * @return 矩阵的副本
     */
    virtual std::unique_ptr<Mat> clone() const = 0;

    /**
     * @brief 重塑矩阵形状
     * @param shape 新的形状
     * @return 重塑后的矩阵
     */
    virtual std::unique_ptr<Mat> reshape(const Shape &shape) const = 0;

    /**
     * @brief 转置矩阵
     * @return 转置后的矩阵
     */
    virtual std::unique_ptr<Mat> transpose() const = 0;

    /**
     * @brief 矩阵加法
     * @param other 另一个矩阵
     * @return 加法结果
     */
    virtual std::unique_ptr<Mat> operator+(const Mat &other) const = 0;

    /**
     * @brief 矩阵减法
     * @param other 另一个矩阵
     * @return 减法结果
     */
    virtual std::unique_ptr<Mat> operator-(const Mat &other) const = 0;

    /**
     * @brief 元素级乘法
     * @param other 另一个矩阵
     * @return 乘法结果
     */
    virtual std::unique_ptr<Mat> operator*(const Mat &other) const = 0;

    /**
     * @brief 矩阵乘法（真正的矩阵乘法）
     * @param other 另一个矩阵
     * @return 矩阵乘法结果
     */
    virtual std::unique_ptr<Mat> matmul(const Mat &other) const = 0;

    /**
     * @brief 矩阵除法
     * @param other 另一个矩阵
     * @return 除法结果
     */
    virtual std::unique_ptr<Mat> operator/(const Mat &other) const = 0;

    /**
     * @brief 标量加法
     * @param scalar 标量值
     * @return 加法结果
     */
    virtual std::unique_ptr<Mat> add_scalar(data_t scalar) const = 0;

    /**
     * @brief 标量乘法
     * @param scalar 标量值
     * @return 乘法结果
     */
    virtual std::unique_ptr<Mat> mul_scalar(data_t scalar) const = 0;

    /**
     * @brief 标量加法运算符
     * @param scalar 标量值
     * @return 加法结果
     */
    virtual std::unique_ptr<Mat> operator+(data_t scalar) const = 0;

    /**
     * @brief 标量减法运算符
     * @param scalar 标量值
     * @return 减法结果
     */
    virtual std::unique_ptr<Mat> operator-(data_t scalar) const = 0;

    /**
     * @brief 标量乘法运算符
     * @param scalar 标量值
     * @return 乘法结果
     */
    virtual std::unique_ptr<Mat> operator*(data_t scalar) const = 0;

    /**
     * @brief 标量除法运算符
     * @param scalar 标量值
     * @return 除法结果
     */
    virtual std::unique_ptr<Mat> operator/(data_t scalar) const = 0;

    /**
     * @brief 一元负号运算符
     * @return 负值结果
     */
    virtual std::unique_ptr<Mat> operator-() const = 0;

    /**
     * @brief 广播到指定形状
     * @param shape 目标形状
     * @return 广播后的矩阵
     */
    virtual std::unique_ptr<Mat> broadcast_to(const Shape &shape) const = 0;

    /**
     * @brief 求和到指定形状
     * @param shape 目标形状
     * @return 求和后的矩阵
     */
    virtual std::unique_ptr<Mat> sum_to(const Shape &shape) const = 0;

    /**
     * @brief 沿指定轴求和
     * @param axis 轴索引，-1表示所有元素
     * @return 求和结果
     */
    virtual std::unique_ptr<Mat> sum(int axis = -1) const = 0;

    /**
     * @brief 获取矩阵形状
     * @return 矩阵形状
     */
    virtual Shape shape() const = 0;

    /**
     * @brief 获取矩阵元素数量
     * @return 元素数量
     */
    virtual size_t elements() const = 0;

    /**
     * @brief 获取标量值（仅适用于单元素矩阵）
     * @return 标量值
     */
    template <typename T>
    T scalar() const;

    /**
     * @brief 打印矩阵内容
     * @param desc 描述信息
     */
    virtual void print(const std::string &desc = "") const = 0;

    /**
     * @brief 转换为向量
     * @return 矩阵数据的向量表示
     */
    virtual std::vector<data_t> to_vector() const = 0;

    // 数学函数
    /**
     * @brief 指数函数
     * @return 指数运算结果
     */
    virtual std::unique_ptr<Mat> exp() const = 0;

    /**
     * @brief 对数函数
     * @return 对数运算结果
     */
    virtual std::unique_ptr<Mat> log() const = 0;

    /**
     * @brief 正弦函数
     * @return 正弦运算结果
     */
    virtual std::unique_ptr<Mat> sin() const = 0;

    /**
     * @brief 余弦函数
     * @return 余弦运算结果
     */
    virtual std::unique_ptr<Mat> cos() const = 0;

    /**
     * @brief 平方根函数
     * @return 平方根运算结果
     */
    virtual std::unique_ptr<Mat> sqrt() const = 0;

    /**
     * @brief 平方函数
     * @return 平方运算结果
     */
    virtual std::unique_ptr<Mat> square() const = 0;

    /**
     * @brief 幂函数
     * @param exponent 指数
     * @return 幂运算结果
     */
    virtual std::unique_ptr<Mat> pow(data_t exponent) const = 0;

    /**
     * @brief 求和
     * @return 所有元素的和
     */
    virtual data_t sum() const = 0;

    /**
     * @brief 求最大值
     * @return 最大值
     */
    virtual data_t max() const = 0;

    /**
     * @brief 求最小值
     * @return 最小值
     */
    virtual data_t min() const = 0;

    /**
     * @brief 求均值
     * @return 均值
     */
    virtual data_t mean() const = 0;
};

/**
 * @brief Mat工厂函数，用于创建Mat对象而不暴露具体后端类型
 * @param data 数据向量
 * @param shape 矩阵形状
 * @return Mat对象的智能指针
 */
std::unique_ptr<Mat> create_mat(const std::vector<data_t> &data, const Shape &shape);

/**
 * @brief Mat工厂函数，用于创建标量矩阵
 * @param value 标量值
 * @param shape 矩阵形状
 * @return Mat对象的智能指针
 */
std::unique_ptr<Mat> create_mat(data_t value, const Shape &shape);

}  // namespace dl

#endif  // __ORIGIN_DL_MAT_H__
