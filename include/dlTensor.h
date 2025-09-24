#ifndef __ORIGIN_DL_TENSOR_H__
#define __ORIGIN_DL_TENSOR_H__

#include "base/dlCommon.h"
#include "dlTensorImpl.h"

namespace dl
{

class Tensor
{
private:
    TensorImplPtr impl_;  // 唯一的成员：智能指针
    
public:
    // 构造函数
    Tensor(const NdArray &data) 
        : impl_(std::make_shared<TensorImpl>(data)) {}
    Tensor(NdArray &&data) 
        : impl_(std::make_shared<TensorImpl>(std::move(data))) {}
    Tensor(TensorImplPtr impl) : impl_(impl) {}
    
    // 拷贝构造函数 - 浅拷贝，共享实现
    Tensor(const Tensor &other) : impl_(other.impl_) {}
    
    // 移动构造函数 - 转移所有权
    Tensor(Tensor &&other) noexcept : impl_(std::move(other.impl_)) {}
    
    // 赋值运算符
    Tensor& operator=(const Tensor &other) {
        impl_ = other.impl_;
        return *this;
    }
    
    Tensor& operator=(Tensor &&other) noexcept {
        impl_ = std::move(other.impl_);
        return *this;
    }
    
    // 析构函数
    ~Tensor() = default;
    
    // 访问器 - 委托给实现
    const NdArray& data() const { return impl_->data_; }
    NdArray& data() { return impl_->data_; }
    
    const NdArray& grad() const { return impl_->grad_; }
    NdArray& grad() { return impl_->grad_; }
    
    // 方法委托
    void set_creator(const FunctionPtr &func) { impl_->set_creator(func); }
    void backward() { impl_->backward(); }
    void clear_grad() { impl_->clear_grad(); }
    
    // 张量操作 - 返回新的 Tensor
    Tensor reshape(const af::dim4 &shape) const {
        return Tensor(std::make_shared<TensorImpl>(impl_->reshape(shape)));
    }
    
    Tensor transpose() const {
        return Tensor(std::make_shared<TensorImpl>(impl_->transpose()));
    }
    
    // 运算符重载已移至 dlOperator.h 中定义
    
    // 调试
    void print(const std::string &desc = "") const {
        impl_->print(desc);
    }
    
    // 内部访问器（用于 Operator）
    TensorImplPtr get_impl() const { return impl_; }
};

// 类型别名
using TensorPtr = std::shared_ptr<Tensor>;

// 获取 shared_ptr 版本（用于计算图管理）
inline TensorPtr get_shared_ptr(const Tensor& tensor) {
    // 创建一个新的 shared_ptr，但共享相同的 impl_
    return std::make_shared<Tensor>(tensor);
}
using TensorList = std::vector<Tensor>;

// 工厂函数
TensorPtr make_tensor(const NdArray &data);
TensorPtr make_tensor(NdArray &&data);

// 全局运算符重载

}  // namespace dl

#endif