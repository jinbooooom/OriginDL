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
 * @details 替代ArrayFire的af::dim4，使用std::vector<size_t>存储维度信息
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
     * @brief 从4个维度构造（兼容ArrayFire的af::dim4）
     * @param d0 第0维
     * @param d1 第1维，默认为1
     * @param d2 第2维，默认为1
     * @param d3 第3维，默认为1
     */
    Shape(size_t d0, size_t d1 = 1, size_t d2 = 1, size_t d3 = 1) : dims_{d0, d1, d2, d3} {}

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
     * @brief 转换为ArrayFire的dim4格式
     * @return af::dim4对象
     */
    af::dim4 to_af_dim4() const
    {
        size_t d0 = dims_.size() > 0 ? dims_[0] : 1;
        size_t d1 = dims_.size() > 1 ? dims_[1] : 1;
        size_t d2 = dims_.size() > 2 ? dims_[2] : 1;
        size_t d3 = dims_.size() > 3 ? dims_[3] : 1;
        return af::dim4(d0, d1, d2, d3);
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
