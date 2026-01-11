#ifndef __ORIGIN_DL_TENSOR_H__
#define __ORIGIN_DL_TENSOR_H__

#include "../common/inner_types.h"
#include "../utils/static_assert.h"
#include "tensor_impl.h"
#include "tensor_options.h"

// 前向声明
class Mat;

namespace origin
{
/*
Tensor 架构层次：
Tensor (用户接口)
    ↓ 只调用TensorImpl方法
TensorImpl (核心实现)
    ↓ 只调用Mat接口方法
Mat (抽象接口)
    ↓ 具体实现
TorchMat/OriginMat (具体后端)

    Tensor (值语义包装)
    └─> std::shared_ptr<TensorImpl>
          └─> data_: shared_ptr<Mat>
                └─> OriginMat::storage_: shared_ptr<Storage>
                      └─> Storage::data_: void* (真正的数据)

          └─> grad_: shared_ptr<Mat>
                └─> OriginMat::storage_: shared_ptr<Storage>
                      └─> Storage::data_: void* (真正的数据)

Storage 是数据拥有者，可以被多个 Mat 共享
Mat 可以是数据拥有者或视图（view）
TensorImpl 的 data_ 和 grad_ 应该直接共享，不需要 clone
reshape 等操作应该创建视图，共享 Storage

数据流路径分析
路径 1: 算子前向传播
用户代码: Tensor y = x0 + x1;
  ↓
Add::forward(xs)
  ↓
mat(xs[0]) + mat(xs[1])  // 返回 unique_ptr<Mat>
  ↓
convert_mat_to_tensor(std::move(result))  // unique_ptr<Mat> -> Tensor
  ↓
Tensor(std::unique_ptr<Mat>)  // Tensor 构造函数
  ↓
TensorImpl(std::unique_ptr<Mat>)  // unique_ptr -> shared_ptr 转换
  ↓
OriginMat 持有新的 Storage (数据已计算)

路径 2: 梯度反向传播
backward()
  ↓
梯度计算: gx = gy * x  // 返回 unique_ptr<Mat>
  ↓
x.impl_->grad_ = gx.impl_->data_  // 共享或累加
  ↓
累加时: x.impl_->grad_->add_inplace(*gx.impl_->data_)  // 原地累加（已优化）

路径 3: reshape/transpose
Tensor::reshape(shape)
  ↓
TensorImpl::reshape(shape)
  ↓
data_->reshape(shape)  // 返回 unique_ptr<Mat>
  ↓
当前实现: 连续张量使用view()（零拷贝），非连续张量创建新Storage（已优化）
*/

/**
 * @brief 张量类，深度学习计算的核心数据结构
 * @details 使用抽象层设计，支持多种后端，完全隐藏底层实现
 */
class Tensor
{
private:
    std::shared_ptr<TensorImpl> impl_;  // 唯一的成员：智能指针

    // 内部构造函数 - 仅限内部使用
    Tensor(std::shared_ptr<TensorImpl> impl);

public:

    Tensor() = default;  // TODO，可以去掉

    Tensor(const Tensor &other);

    Tensor(Tensor &&other) noexcept;

    Tensor &operator=(const Tensor &other);

    Tensor &operator=(Tensor &&other) noexcept;

    ~Tensor() = default;

    // 向量构造函数（自动推断类型）
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape)
        : Tensor(data, shape, DataTypeTraits<T>::type)  // 根据T推断数据类型，然后委托给DataType版本的构造函数
    {
        // 在编译时做静态检查，避免运行时出现问题
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 向量构造函数（指定数据类型）
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape, DataType dtype)
        : Tensor(data, shape, TensorOptions(dtype))  // 委托给TensorOptions版本的构造函数
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 向量构造函数（指定TensorOptions）
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape, const TensorOptions &options)
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);

        // 验证数据大小与形状是否匹配
        size_t expected_elements = shape.elements();
        if (data.size() != expected_elements)
        {
            throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                        ") does not match shape elements (" + std::to_string(expected_elements) + ")");
        }

        from_memory(data.data(), DataTypeTraits<T>::type, shape, options);
    }

    // TODO: 初始化列表的方式到vector的方式有性能问题，未来需要优化
    // 初始化列表构造函数（自动推断类型）
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape)
        : Tensor(std::vector<T>(data), shape)  // 委托给vector版本的构造函数
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 初始化列表构造函数（指定数据类型）
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape, DataType dtype)
        : Tensor(std::vector<T>(data), shape, dtype)  // 委托给vector版本的构造函数
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 初始化列表构造函数（指定TensorOptions）
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape, const TensorOptions &options)
        : Tensor(std::vector<T>(data), shape, options)  // 委托给vector版本的TensorOptions构造函数
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 标量构造函数（自动推断类型）
    template <typename T>
    Tensor(T scalar, const Shape &shape)
        : Tensor(scalar, shape, DataTypeTraits<T>::type)  // 委托给DataType版本的构造函数
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 标量构造函数（指定数据类型）
    template <typename T>
    Tensor(T scalar, const Shape &shape, DataType dtype)
        : Tensor(scalar, shape, TensorOptions(dtype))  // 委托给TensorOptions版本的构造函数
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
    }

    // 标量构造函数（指定TensorOptions）
    template <typename T>
    Tensor(T scalar, const Shape &shape, const TensorOptions &options)
    {
        ORIGIN_STATIC_ASSERT_ARITHMETIC(T);
        from_scalar(scalar, shape, options);
    }

    Tensor(const Scalar &scalar, const Shape &shape, const TensorOptions &options)
    {
        impl_ = std::make_unique<TensorImpl>(TensorImpl::from_scalar(scalar, shape, options));
    }

    // === 工厂方法（只保留TensorOptions版本）===
    static Tensor zeros(const Shape &shape, const TensorOptions &options = TensorOptions());
    static Tensor ones(const Shape &shape, const TensorOptions &options = TensorOptions());
    static Tensor randn(const Shape &shape, const TensorOptions &options = TensorOptions());
    static Tensor full(const Shape &shape, const Scalar &value, const TensorOptions &options = TensorOptions());
    // data 的类型一定要与 options.dtypa()一致，不然在解引用指针data的时候可能会出现问题。
    static Tensor from_blob(void *data, const Shape &shape, const TensorOptions &options = TensorOptions());

    // === 形状和维度 ===
    Shape shape() const;
    size_t ndim() const;
    size_t elements() const;

    // === 张量属性方法（对应PyTorch的tensor.element_size()等） ===
    size_t element_size() const;  // 对应tensor.element_size()
    size_t numel() const;         // 对应tensor.numel()
    size_t nbytes() const;        // 对应tensor.nbytes

    // === 数据访问：类型安全 ===
    template <typename T>
    T item() const;

    template <typename T>
    T *data_ptr();

    // === 类型查询和转换 ===
    DataType dtype() const;
    Device device() const;
    Tensor to(DataType target_type) const;
    Tensor to(Device device) const;
    Tensor to(const TensorOptions &options) const;

    // === 梯度相关 ===
    Tensor grad() const;
    void set_creator(const std::shared_ptr<Operator> &func);
    void backward();
    void clear_grad();
    void accumulate_grad(const Tensor &grad_to_add);

    /**
     * @brief 检查 tensor 是否需要梯度计算
     * @return 如果 tensor 在计算图中（有 creator_）且全局反向传播启用，返回 true
     * @details 类似 PyTorch 的 tensor.requires_grad 属性
     */
    bool requires_grad() const;

    // === 计算图管理 ===
    /**
     * @brief 断开tensor与计算图的连接，创建一个不参与梯度计算的新tensor
     * @return 新的tensor，与原始tensor共享数据但不参与梯度计算
     * @details 类似于PyTorch的detach()方法，用于显式断开计算图，帮助释放内存
     */
    Tensor detach() const;

    // === 张量操作 ===
    Tensor reshape(const Shape &shape) const;
    Tensor transpose() const;

    // === 泛型标量操作 ===
    // 注意：标量操作使用全局操作符重载，避免与成员操作符冲突

    // === 调试 ===
    void print(const std::string &desc = "") const;
    template <typename T>
    std::vector<T> to_vector() const;
    int backend_type() const;

    // 友元类声明
    friend class Operator;
    friend class TensorImpl;

private:
    // 从Mat创建Tensor的构造函数 - 仅限友元类使用
    Tensor(std::unique_ptr<Mat> mat);

    // 友元函数，供内部使用
    friend Tensor create_tensor_from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options);

    // === 用于显式类型指定的方法 ===
    /**
     * @brief 从标量数据创建张量
     * @param scalar 标量值
     * @param shape 张量形状
     * @param options 张量选项
     */
    void from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options);

    /**
     * @brief 从原始数据创建张量
     * @param data 原始数据的指针
     * @param user_dtype 原始数据类型
     * @param shape 张量形状
     * @param options 张量选项
     */
    void from_memory(const void *data, DataType user_dtype, const Shape &shape, const TensorOptions &options);
};

}  // namespace origin

#endif