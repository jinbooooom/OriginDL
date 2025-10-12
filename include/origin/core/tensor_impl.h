#ifndef __ORIGIN_DL_TENSOR_IMPL_H__
#define __ORIGIN_DL_TENSOR_IMPL_H__

#include <memory>
#include <vector>
#include "../common/inner_types.h"
#include "../mat/basic_types.h"
#include "../mat/mat.h"
#include "../mat/shape.h"

// 前向声明
class Operator;

namespace origin
{
/*
为了使 Tensor 看起来像是值语义的指针，把Tensor中所有的数据下沉到 TensorImpl,
Tensor中只保留一个智能指针，方便值传递。

TensorImpl 是 Tensor 的实现类，负责管理底层数据和操作。
它封装了 Mat 抽象层，提供了对底层数据的操作接口。
它还实现了反向传播算法，用于计算梯度。
然后调用底层矩阵计算后端，实现张量操作。
*/
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

    // 支持多种数据类型的构造函数
    template <typename T>
    TensorImpl(const std::vector<T> &data, const Shape &shape)
    {
        auto inferred_type = get_data_type_from_template<T>();
        create_impl_from_data(data.data(), shape, inferred_type);
    }

    template <typename T>
    TensorImpl(T scalar, const Shape &shape)
    {
        auto inferred_type = get_data_type_from_template<T>();
        create_impl_from_scalar(scalar, shape, inferred_type);
    }

    // 从void*数据构造，需要指定数据类型
    TensorImpl(const void *data, const Shape &shape, DataType dtype);

    // 静态工厂方法
    static TensorImpl randn(const Shape &shape);

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
    template <typename T>
    TensorImpl operator+(T scalar) const;
    TensorImpl operator-(const TensorImpl &other) const;
    TensorImpl operator*(const TensorImpl &other) const;
    TensorImpl operator/(const TensorImpl &other) const;

    // 一元负号运算符
    TensorImpl operator-() const;

    // 访问器方法
    Shape shape() const;
    size_t ndim() const;
    size_t elements() const;
    template <typename T>
    T item() const;
    template <typename T>
    std::vector<T> to_vector() const;
    int backend_type() const;

    // === 泛型数据访问方法 ===
    template <typename T>
    T *data_ptr();

    template <typename T>
    TensorImpl operator-(T scalar) const;

    template <typename T>
    TensorImpl operator*(T scalar) const;

    template <typename T>
    TensorImpl operator/(T scalar) const;

    // 调试
    void print(const std::string &desc = "") const;

private:
    // 类型推断辅助函数
    template <typename T>
    DataType get_data_type()
    {
        return get_data_type_from_template<T>();
    }

    // 张量创建辅助函数
    void create_impl_from_data(const void *data, const Shape &shape, DataType dtype);
    void create_impl_from_scalar(double data, const Shape &shape, DataType dtype);

    // 模板化的张量创建实现 - 声明在头文件，实现在cpp文件
    template <typename T>
    void create_impl_impl(const T *data, size_t count, const Shape &shape);

    template <typename T>
    void create_impl_impl(T scalar, const Shape &shape);
};

using TensorImplPtr = std::shared_ptr<TensorImpl>;

}  // namespace origin

#endif
