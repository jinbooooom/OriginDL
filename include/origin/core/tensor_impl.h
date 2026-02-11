#ifndef __ORIGIN_DL_TENSOR_IMPL_H__
#define __ORIGIN_DL_TENSOR_IMPL_H__

#include <memory>
#include <vector>
#include "../common/inner_types.h"
#include "../mat/basic_types.h"
#include "../mat/mat.h"
#include "../mat/shape.h"
#include "tensor_options.h"

namespace origin
{
// 前向声明
class Operator;
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
    std::shared_ptr<Mat> data_;  // 使用Mat抽象层，支持共享（与PyTorch行为一致）
    std::shared_ptr<Mat> grad_;  // 使用Mat抽象层，支持共享（与PyTorch行为一致）
    std::shared_ptr<Operator> creator_;
    int generation_;

    // 核心构造函数 - 接受 unique_ptr 并转换为 shared_ptr（底层返回 unique_ptr，表示数据所有权转移）
    TensorImpl(std::unique_ptr<Mat> data)
        : data_(std::shared_ptr<Mat>(std::move(data))), grad_(nullptr), creator_(nullptr), generation_(0)
    {}

    // 核心构造函数 - 接受 shared_ptr（用于内部共享）
    TensorImpl(std::shared_ptr<Mat> data) : data_(data), grad_(nullptr), creator_(nullptr), generation_(0) {}

    // 两个核心工厂方法
    static TensorImpl from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options);
    static TensorImpl from_memory(const void *data,
                                  DataType user_dtype,
                                  const Shape &shape,
                                  const TensorOptions &options);

    // 静态工厂方法
    static TensorImpl randn(const Shape &shape);
    static TensorImpl randn(const Shape &shape, const TensorOptions &options);

    // 拷贝构造函数 - clone data_ 和 grad_（保证值语义，拷贝后独立）
    TensorImpl(const TensorImpl &other)
        : data_(other.data_ ? std::shared_ptr<Mat>(other.data_->clone()) : nullptr),
          grad_(other.grad_ ? std::shared_ptr<Mat>(other.grad_->clone()) : nullptr),
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
    void set_creator(const std::shared_ptr<Operator> &func);
    void backward();
    void clear_grad();

    // 计算图管理
    /**
     * @brief 断开tensor与计算图的连接
     * @details 将creator_设置为nullptr，断开与Operator的连接，帮助释放计算图内存
     */
    void detach();

    // 张量操作
    TensorImpl reshape(const Shape &shape) const;
    TensorImpl transpose() const;

    // 运算符重载 - 一元负号运算符已移除，通过算子层实现

    // 访问器方法
    Shape shape() const;
    size_t ndim() const;
    size_t elements() const;
    template <typename T>
    T item() const;
    int backend_type() const;

    // === 泛型数据访问方法 ===
    template <typename T>
    T *data_ptr();

    // === 索引访问 ===
    /**
     * @brief 根据多维索引读取单个元素
     * @param indices 多维索引，例如 {i, j, k} 表示访问 tensor[i][j][k]
     * @return 索引位置的值
     */

    Scalar index(std::initializer_list<size_t> indices) const;

    /**
     * @brief 根据多维索引写入单个元素
     * @param indices 多维索引，例如 {i, j, k} 表示访问 tensor[i][j][k]
     * @param value 要写入的标量值，会自动转换为与tensor相同的数据类型
     */
    void index_put(std::initializer_list<size_t> indices, const Scalar &value);

    // 调试
    void print(const std::string &desc = "") const;

    // 类型转换
    TensorImpl to(const TensorOptions &options) const;
};

}  // namespace origin

#endif
