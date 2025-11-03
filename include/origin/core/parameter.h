#ifndef __ORIGIN_DL_PARAMETER_H__
#define __ORIGIN_DL_PARAMETER_H__

#include "tensor.h"

namespace origin
{

/**
 * @brief 模型参数类，继承自Tensor
 * @details Parameter用于标识需要优化的模型参数，与普通Tensor区分
 */
class Parameter : public Tensor
{
public:
    // 继承Tensor的所有构造函数
    using Tensor::Tensor;

    // 从Tensor构造（允许Tensor转换为Parameter）
    Parameter(const Tensor &tensor) : Tensor(tensor) {}

    // 从Tensor移动构造
    Parameter(Tensor &&tensor) : Tensor(std::move(tensor)) {}

    // 默认构造函数
    Parameter() = default;

    // 拷贝构造函数
    Parameter(const Parameter &other) = default;

    // 移动构造函数
    Parameter(Parameter &&other) noexcept = default;

    // 赋值运算符
    Parameter &operator=(const Parameter &other)     = default;
    Parameter &operator=(Parameter &&other) noexcept = default;

    // 从Tensor赋值
    Parameter &operator=(const Tensor &other)
    {
        static_cast<Tensor &>(*this) = other;
        return *this;
    }

    Parameter &operator=(Tensor &&other)
    {
        static_cast<Tensor &>(*this) = std::move(other);
        return *this;
    }

    // 析构函数
    ~Parameter() = default;
};

}  // namespace origin

#endif  // __ORIGIN_DL_PARAMETER_H__
