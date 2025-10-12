#ifndef __ORIGIN_DL_MAT_H__
#define __ORIGIN_DL_MAT_H__

#include <memory>
#include <vector>
#include "basic_types.h"
#include "shape.h"

namespace origin
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

    // === 泛型标量操作 ===
    template <typename T>
    std::unique_ptr<Mat> add_scalar(T scalar) const;

    template <typename T>
    std::unique_ptr<Mat> mul_scalar(T scalar) const;

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

    // === 泛型标量运算符 ===
    template <typename T>
    std::unique_ptr<Mat> operator+(T scalar) const;

    template <typename T>
    std::unique_ptr<Mat> operator-(T scalar) const;

    template <typename T>
    std::unique_ptr<Mat> operator*(T scalar) const;

    template <typename T>
    std::unique_ptr<Mat> operator/(T scalar) const;

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

    template <typename T>
    T *data_ptr();

    template <typename T>
    std::vector<T> to_vector() const;

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
     * @brief 求和 - 返回标量值
     * @return 所有元素的和
     * @note 重命名为 sum_all() 是为了避免与 sum(int axis) 函数名冲突
     *       原来命名为 sum() 会导致编译器无法区分按轴求和和全局求和
     */
    virtual data_t sum_all() const = 0;

    /**
     * @brief 求最大值 - 返回标量值
     * @return 最大值
     * @note 重命名为 max_all() 是为了保持命名一致性，避免与可能的按轴操作冲突
     */
    virtual data_t max_all() const = 0;

    /**
     * @brief 求最小值 - 返回标量值
     * @return 最小值
     * @note 重命名为 min_all() 是为了保持命名一致性，避免与可能的按轴操作冲突
     */
    virtual data_t min_all() const = 0;

    /**
     * @brief 求均值 - 返回标量值
     * @return 均值
     * @note 重命名为 mean_all() 是为了保持命名一致性，避免与可能的按轴操作冲突
     */
    virtual data_t mean_all() const = 0;

    // === 泛型版本的数据访问方法 ===
    template <typename T>
    T sum_as() const;

    template <typename T>
    T max_as() const;

    template <typename T>
    T min_as() const;

    template <typename T>
    T mean_as() const;

    /**
     * @brief 获取后端类型
     * @return 后端类型标识符
     */
    virtual int backend_type() const = 0;

    /**
     * @brief 获取数据类型
     * @return 数据类型枚举
     */
    virtual DataType dtype() const = 0;

    /**
     * @brief 类型转换
     * @param target_type 目标数据类型
     * @return 转换后的矩阵
     */
    virtual std::unique_ptr<Mat> to(DataType target_type) const = 0;
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

// === Mat模板方法实现 ===
template <typename T>
std::unique_ptr<Mat> Mat::add_scalar(T scalar) const
{
    // 将泛型标量转换为data_t，然后调用现有的虚函数
    return add_scalar(static_cast<data_t>(scalar));
}

template <typename T>
std::unique_ptr<Mat> Mat::mul_scalar(T scalar) const
{
    // 将泛型标量转换为data_t，然后调用现有的虚函数
    return mul_scalar(static_cast<data_t>(scalar));
}

template <typename T>
std::unique_ptr<Mat> Mat::operator+(T scalar) const
{
    return add_scalar<T>(scalar);
}

template <typename T>
std::unique_ptr<Mat> Mat::operator-(T scalar) const
{
    return operator+(-scalar);
}

template <typename T>
std::unique_ptr<Mat> Mat::operator*(T scalar) const
{
    return mul_scalar<T>(scalar);
}

template <typename T>
std::unique_ptr<Mat> Mat::operator/(T scalar) const
{
    return mul_scalar<T>(1.0 / scalar);
}

template <typename T>
T Mat::sum_as() const
{
    // 调用现有的虚函数，然后转换类型
    return static_cast<T>(sum_all());
}

template <typename T>
T Mat::max_as() const
{
    return static_cast<T>(max_all());
}

template <typename T>
T Mat::min_as() const
{
    return static_cast<T>(min_all());
}

template <typename T>
T Mat::mean_as() const
{
    return static_cast<T>(mean_all());
}

template <typename T>
T *Mat::data_ptr()
{
    // 对于泛型data_ptr，我们需要在子类中实现
    // 这里先返回nullptr，子类需要重写这个方法
    return nullptr;
}

template <typename T>
std::vector<T> Mat::to_vector() const
{
    // 调用现有的虚函数，然后转换类型
    auto data_t_vec = to_vector();
    std::vector<T> result;
    result.reserve(data_t_vec.size());
    for (const auto &val : data_t_vec)
    {
        result.push_back(static_cast<T>(val));
    }
    return result;
}

}  // namespace origin

#endif  // __ORIGIN_DL_MAT_H__
