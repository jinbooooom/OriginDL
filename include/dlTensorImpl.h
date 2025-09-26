#ifndef __ORIGIN_DL_TENSOR_IMPL_H__
#define __ORIGIN_DL_TENSOR_IMPL_H__

#include "base/dlCommon.h"

// 前向声明
class Operator;

namespace dl
{

class TensorImpl
{
public:
    std::unique_ptr<Mat> data_;  // 使用Mat抽象层
    std::unique_ptr<Mat> grad_;  // 使用Mat抽象层
    FunctionPtr creator_;
    int generation_;

    // 构造函数
    TensorImpl(std::unique_ptr<Mat> data) : data_(std::move(data)), grad_(nullptr), creator_(nullptr), generation_(0) {}
    TensorImpl(const Mat &data) : data_(data.clone()), grad_(nullptr), creator_(nullptr), generation_(0) {}

    // 拷贝构造函数
    TensorImpl(const TensorImpl &other)
        : data_(other.data_ ? other.data_->clone() : nullptr),
          grad_(other.grad_ ? other.grad_->clone() : nullptr),
          creator_(other.creator_),
          generation_(other.generation_)
    {}

    // 移动构造函数
    TensorImpl(TensorImpl &&other) noexcept
        : data_(std::move(other.data_)),
          grad_(std::move(other.grad_)),
          creator_(std::move(other.creator_)),
          generation_(other.generation_)
    {}

    // 赋值运算符
    TensorImpl &operator=(const TensorImpl &other);
    TensorImpl &operator=(TensorImpl &&other) noexcept;

    // 析构函数
    ~TensorImpl() = default;

    // 核心方法
    void set_creator(const FunctionPtr &func);
    void backward();
    void clear_grad();

    // 张量操作
    TensorImpl reshape(const Shape &shape) const;
    TensorImpl transpose() const;

    // 运算符重载
    TensorImpl operator+(const TensorImpl &other) const;
    TensorImpl operator+(data_t scalar) const;
    TensorImpl operator-(const TensorImpl &other) const;
    TensorImpl operator*(const TensorImpl &other) const;
    TensorImpl operator/(const TensorImpl &other) const;

    // 一元负号运算符
    TensorImpl operator-() const;

    // 调试
    void print(const std::string &desc = "") const;
};

using TensorImplPtr = std::shared_ptr<TensorImpl>;

}  // namespace dl

#endif
