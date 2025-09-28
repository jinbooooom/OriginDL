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

    // 测试用公共访问器（仅限测试使用）
    const Mat &data_for_test() const { return *impl_->data_; }
    Mat &data_for_test() { return *impl_->data_; }
    
    // 测试用构造函数（仅限测试使用）
    static Tensor from_mat_for_test(std::unique_ptr<Mat> mat) { return Tensor(std::move(mat)); }

    // 友元类声明
    friend class Operator;
    friend class TensorImpl;

private:
    // 从Mat创建Tensor的构造函数 - 仅限友元类使用
    Tensor(std::unique_ptr<Mat> mat) : impl_(std::make_shared<TensorImpl>(std::move(mat))) {}


};

// 类型别名
using TensorPtr = std::shared_ptr<Tensor>;

// 获取 shared_ptr 版本（用于计算图管理）
inline TensorPtr get_shared_ptr(const Tensor &tensor)
{
    // 创建一个新的 shared_ptr，但共享相同的 impl_
    return std::make_shared<Tensor>(tensor);
}
using TensorList = std::vector<Tensor>;

}  // namespace dl

#endif