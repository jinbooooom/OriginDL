#ifndef __ORIGIN_DL_TENSOR_IMPL_H__
#define __ORIGIN_DL_TENSOR_IMPL_H__

#include "base/dlCommon.h"

namespace dl
{

class TensorImpl
{
public:
    NdArray data_;
    NdArray grad_;
    FunctionPtr creator_;
    int generation_;

    // 构造函数
    TensorImpl(const NdArray &data) 
        : data_(data), grad_(), creator_(nullptr), generation_(0) {}
    TensorImpl(NdArray &&data) 
        : data_(std::move(data)), grad_(), creator_(nullptr), generation_(0) {}
    
    // 拷贝构造函数
    TensorImpl(const TensorImpl &other) 
        : data_(other.data_), grad_(other.grad_), creator_(other.creator_), generation_(other.generation_) {}
    
    // 移动构造函数
    TensorImpl(TensorImpl &&other) noexcept 
        : data_(std::move(other.data_)), 
          grad_(std::move(other.grad_)), 
          creator_(std::move(other.creator_)), 
          generation_(other.generation_) {}
    
    // 赋值运算符
    TensorImpl& operator=(const TensorImpl &other);
    TensorImpl& operator=(TensorImpl &&other) noexcept;
    
    // 析构函数
    ~TensorImpl() = default;

    // 核心方法
    void set_creator(const FunctionPtr &func);
    void backward();
    void clear_grad();
    
    // 张量操作
    TensorImpl reshape(const af::dim4 &shape) const;
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
