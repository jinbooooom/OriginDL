#ifndef __ORIGIN_DL_TENSOR_H__
#define __ORIGIN_DL_TENSOR_H__

#include "../common/inner_types.h"
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
*/

/**
 * @brief 张量类，深度学习计算的核心数据结构
 * @details 使用抽象层设计，支持多种后端，完全隐藏底层实现
 */
class Tensor
{
private:
    TensorImplPtr impl_;  // 唯一的成员：智能指针

    // 内部构造函数 - 仅限内部使用
    Tensor(TensorImplPtr impl);

public:
    // 默认构造函数
    Tensor() = default;  // TODO，可以去掉
    // 拷贝构造函数 - 浅拷贝，共享实现
    Tensor(const Tensor &other);
    // 移动构造函数 - 转移所有权
    Tensor(Tensor &&other) noexcept;
    // 赋值运算符
    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;
    // 析构函数
    ~Tensor() = default;

    // 向量构造函数（自动推断类型）
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape)
        : Tensor(data, shape, get_data_type_from_template<T>())  // 根据T推断数据类型，然后委托给DataType版本的构造函数
    {}

    // 向量构造函数（指定数据类型）
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape, DataType dtype)
        : Tensor(data, shape, TensorOptions(dtype))  // 委托给TensorOptions版本的构造函数
    {}

    // 向量构造函数（指定TensorOptions）
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape, const TensorOptions &options)
    {
        // 验证数据大小与形状是否匹配
        size_t expected_elements = shape.elements();
        if (data.size() != expected_elements)
        {
            throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                        ") does not match shape elements (" + std::to_string(expected_elements) + ")");
        }

        create_tensor_from_data_with_dtype(data.data(), data.size(), shape, options.dtype());
        // 如果设备不是CPU，需要移动到指定设备
        if (options.device().type() != DeviceType::kCPU)
        {
            impl_ = std::make_shared<TensorImpl>(impl_->to(options));
        }
    }

    // TODO: 初始化列表的方式到vector的方式有性能问题，未来需要优化
    // 初始化列表构造函数（自动推断类型）
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape)
        : Tensor(std::vector<T>(data), shape)  // 委托给vector版本的构造函数
    {}

    // 初始化列表构造函数（指定数据类型）
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape, DataType dtype)
        : Tensor(std::vector<T>(data), shape, dtype)  // 委托给vector版本的构造函数
    {}

    // 初始化列表构造函数（指定TensorOptions）
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape, const TensorOptions &options)
        : Tensor(std::vector<T>(data), shape, options)  // 委托给vector版本的TensorOptions构造函数
    {}

    // 标量构造函数（自动推断类型）
    template <typename T>
    Tensor(T scalar, const Shape &shape)
        : Tensor(scalar, shape, get_data_type_from_template<T>())  // 委托给DataType版本的构造函数
    {}

    // 标量构造函数（指定数据类型）
    template <typename T>
    Tensor(T scalar, const Shape &shape, DataType dtype)
        : Tensor(scalar, shape, TensorOptions(dtype))  // 委托给TensorOptions版本的构造函数
    {}

    // 标量构造函数（指定TensorOptions）
    template <typename T>
    Tensor(T scalar, const Shape &shape, const TensorOptions &options)
    {
        static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type (int, float, double, etc.)");
        static_assert(!std::is_pointer_v<T>, "T cannot be a pointer type");
        create_tensor_from_scalar_with_dtype(scalar, shape, options.dtype());
        // 如果设备不是CPU，需要移动到指定设备
        if (options.device().type() != DeviceType::kCPU)
        {
            impl_ = std::make_shared<TensorImpl>(impl_->to(options));
        }
    }

    // === 工厂方法（只保留TensorOptions版本）===
    static Tensor zeros(const Shape &shape, const TensorOptions &options = TensorOptions());
    static Tensor ones(const Shape &shape, const TensorOptions &options = TensorOptions());
    static Tensor randn(const Shape &shape, const TensorOptions &options = TensorOptions());
    static Tensor full(const Shape &shape, double value, const TensorOptions &options = TensorOptions());
    static Tensor from_blob(void *data, const Shape &shape, const TensorOptions &options = TensorOptions());

    // === 形状和维度 ===
    Shape shape() const;
    size_t ndim() const;
    size_t elements() const;

    // === 数据访问：类型安全 ===
    template <typename T>
    T item() const;

    template <typename T>
    T *data_ptr();

    // === 类型查询和转换 ===
    DataType dtype() const;
    Tensor to(DataType target_type) const;
    Tensor to(const TensorOptions &options) const;

    // === 梯度相关 ===
    Tensor grad() const;
    void set_creator(const FunctionPtr &func);
    void backward();
    void clear_grad();

    // === 张量操作 ===
    Tensor reshape(const Shape &shape) const;
    Tensor transpose() const;

    // === 泛型标量操作 ===
    template <typename T>
    Tensor operator+(T scalar) const;

    template <typename T>
    Tensor operator-(T scalar) const;

    template <typename T>
    Tensor operator*(T scalar) const;

    template <typename T>
    Tensor operator/(T scalar) const;

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

    // === 内部辅助方法 ===
    template <typename T>
    DataType get_data_type()
    {
        return get_data_type_from_template<T>();
    }

    // === 用于显式类型指定的方法 ===
    /**
     * @brief 从标量数据创建张量（显式指定类型）
     * @details 用于工厂方法如full()，标量数据是单个值，用void*传递，类型信息由dtype参数提供
     * @param data 标量数据的指针
     * @param shape 张量形状
     * @param dtype 目标数据类型
     */
    template <typename T>
    void create_tensor_from_scalar_with_dtype(T scalar, const Shape &shape, DataType dtype);

    /**
     * @brief 从原始数据创建张量（显式指定类型）
     * @details 用于from_blob()方法，原始数据是void*指针，直接传递，类型信息由dtype参数提供
     * @param data 原始数据的指针
     * @param shape 张量形状
     * @param dtype 目标数据类型
     */
    void create_tensor_from_raw_data(const void *data, const Shape &shape, DataType dtype);

    /**
     * @brief 从带类型的数据创建张量（显式指定类型）
     * @details 用于带DataType的构造函数，带类型的数据保持原始类型信息，可以进行类型转换和验证
     * @param data 带类型数据的指针
     * @param count 数据元素数量
     * @param shape 张量形状
     * @param dtype 目标数据类型
     */
    template <typename T>
    void create_tensor_from_data_with_dtype(const T *data, size_t count, const Shape &shape, DataType dtype);
};

}  // namespace origin

#endif