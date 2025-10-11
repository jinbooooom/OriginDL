#ifndef __ORIGIN_DL_SHAPE_H__
#define __ORIGIN_DL_SHAPE_H__

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <vector>

namespace origin
{

/**
 * @brief 张量形状类，用于表示多维张量的维度信息
 * @details 替代底层矩阵计算库的形状信息，使用std::vector<size_t>存储维度信息
 */
class Shape
{
private:
    std::vector<size_t> dims_;

public:
    /**
     * @brief 默认构造函数
     */
    Shape() = default;

    /**
     * @brief 从维度向量构造
     * @param dims 维度向量
     */
    Shape(const std::vector<size_t> &dims) : dims_(dims) {}

    /**
     * @brief 从初始化列表构造
     * @param dims 维度列表
     */
    Shape(std::initializer_list<size_t> dims) : dims_(dims) {}


    /**
     * @brief 获取维度向量
     * @return 维度向量的常量引用
     */
    const std::vector<size_t> &dims() const { return dims_; }

    /**
     * @brief 获取维度数量
     * @return 维度数量
     */
    size_t size() const { return dims_.size(); }

    /**
     * @brief 获取维度数量（别名）
     * @return 维度数量
     */
    size_t ndims() const { return dims_.size(); }

    /**
     * @brief 访问指定维度的值
     * @param i 维度索引
     * @return 维度值
     * @throws std::out_of_range 如果索引超出范围
     */
    size_t operator[](size_t i) const
    {
        if (i >= dims_.size())
        {
            throw std::out_of_range("Shape index " + std::to_string(i) + " out of range for size " +
                                    std::to_string(dims_.size()));
        }
        return dims_[i];
    }

    /**
     * @brief 访问指定维度的值（可修改）
     * @param i 维度索引
     * @return 维度值的引用
     * @throws std::out_of_range 如果索引超出范围
     */
    size_t &operator[](size_t i)
    {
        if (i >= dims_.size())
        {
            throw std::out_of_range("Shape index " + std::to_string(i) + " out of range for size " +
                                    std::to_string(dims_.size()));
        }
        return dims_[i];
    }

    /**
     * @brief 计算总元素数量
     * @return 所有维度的乘积
     * @throws std::overflow_error 如果计算结果溢出
     */
    size_t elements() const
    {
        size_t result = 1;
        for (size_t dim : dims_)
        {
            // 检查乘法溢出
            if (dim != 0 && result > SIZE_MAX / dim)
            {
                throw std::overflow_error("Shape elements calculation overflow: result would exceed SIZE_MAX " +
                                          std::to_string(SIZE_MAX));
            }
            result *= dim;
        }
        return result;
    }

    /**
     * @brief 比较操作
     * @param other 另一个Shape对象
     * @return 是否相等
     */
    bool operator==(const Shape &other) const { return dims_ == other.dims_; }

    /**
     * @brief 比较操作
     * @param other 另一个Shape对象
     * @return 是否不相等
     */
    bool operator!=(const Shape &other) const { return dims_ != other.dims_; }

    /**
     * @brief 转换为字符串
     * @return 形状的字符串表示
     */
    std::string to_string() const
    {
        std::string result = "[";
        for (size_t i = 0; i < dims_.size(); ++i)
        {
            if (i > 0)
                result += ", ";
            result += std::to_string(dims_[i]);
        }
        result += "]";
        return result;
    }

    /**
     * @brief 输出操作符
     * @param os 输出流
     * @param shape Shape对象
     * @return 输出流引用
     */
    friend std::ostream &operator<<(std::ostream &os, const Shape &shape)
    {
        os << "[";
        for (size_t i = 0; i < shape.dims_.size(); ++i)
        {
            if (i > 0)
                os << ", ";
            os << shape.dims_[i];
        }
        os << "]";
        return os;
    }

    // 迭代器支持
    auto begin() const { return dims_.begin(); }
    auto end() const { return dims_.end(); }
    auto begin() { return dims_.begin(); }
    auto end() { return dims_.end(); }
};

}  // namespace origin

#endif  // __ORIGIN_DL_SHAPE_H__
