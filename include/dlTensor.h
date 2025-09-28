#ifndef __ORIGIN_DL_TENSOR_H__
#define __ORIGIN_DL_TENSOR_H__

#include "base/dlCommon.h"
#include "dlTensorImpl.h"

namespace dl
{

// 前向声明
class Operator;

/**
 * @brief 张量类，深度学习计算的核心数据结构
 * @details 使用抽象层设计，支持多种后端，完全隐藏底层实现
 */
class Tensor
{
private:
    TensorImplPtr impl_;  // 唯一的成员：智能指针

    // 内部构造函数 - 仅限内部使用
    Tensor(TensorImplPtr impl) : impl_(impl) {}

public:
    // 公共构造函数
    Tensor(const std::vector<data_t> &data, const Shape &shape);
    Tensor(std::initializer_list<data_t> data, const Shape &shape);
    Tensor(data_t scalar, const Shape &shape);

    // 拷贝构造函数 - 浅拷贝，共享实现
    Tensor(const Tensor &other) : impl_(other.impl_) {}

    // 移动构造函数 - 转移所有权
    Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}


    // 赋值运算符
    Tensor &operator=(const Tensor &other)
    {
        impl_ = other.impl_;
        return *this;
    }

    Tensor &operator=(Tensor &&other) noexcept
    {
        impl_ = std::move(other.impl_);
        return *this;
    }

    // 析构函数
    ~Tensor() = default;

    // 工厂函数
    static Tensor zeros(const Shape &shape);
    static Tensor ones(const Shape &shape);
    static Tensor randn(const Shape &shape);
    
    // /**
    //  * @brief 生成均匀分布的随机张量
    //  * @param shape 张量形状
    //  * @return 值在[0,1)范围内的随机张量
    //  * @details 使用均匀分布生成随机数，适用于测试数据生成
    //  */
    // static Tensor randu(const Shape &shape);
    
    // /**
    //  * @brief 生成序列张量
    //  * @param shape 张量形状
    //  * @return 从0开始的连续整数序列张量
    //  * @details 生成形如[0,1,2,3,...]的序列，按列优先顺序填充
    //  * @example
    //  * Tensor t = Tensor::iota(Shape{3, 4});
    //  * 生成矩阵:
    //  * 0.0000  3.0000  6.0000  9.0000
    //  * 1.0000  4.0000  7.0000 10.0000  
    //  * 2.0000  5.0000  8.0000 11.0000
    //  */
    // static Tensor iota(const Shape &shape);
    
    static Tensor constant(data_t value, const Shape &shape);
    static Tensor from_data(const std::vector<data_t> &data, const Shape &shape);

    // 公共访问器
    Shape shape() const;
    size_t ndim() const;
    size_t elements() const;

    // 标量访问
    data_t item() const;

    // 梯度访问
    Tensor grad() const;

    // 方法委托
    void set_creator(const FunctionPtr &func) { impl_->set_creator(func); }
    void backward() { impl_->backward(); }
    void clear_grad() { impl_->clear_grad(); }

    // 张量操作 - 返回新的 Tensor
    Tensor reshape(const Shape &shape) const;
    Tensor transpose() const;

    // 数据转换
    std::vector<data_t> to_vector() const;

    // 调试
    void print(const std::string &desc = "") const { impl_->print(desc); }

    // 友元类声明
    friend class Operator;
    friend class TensorImpl;

private:
    // 从Mat创建Tensor的构造函数 - 仅限友元类使用
    Tensor(std::unique_ptr<Mat> mat) : impl_(std::make_shared<TensorImpl>(std::move(mat))) {}


};

}  // namespace dl

#endif