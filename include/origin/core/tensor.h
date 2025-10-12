#ifndef __ORIGIN_DL_TENSOR_H__
#define __ORIGIN_DL_TENSOR_H__

#include "../common/inner_types.h"
#include "tensor_impl.h"

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
TorchMat (具体后端)
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
    Tensor() = default;

    // === 构造函数：自动类型推断 ===
    template <typename T>
    Tensor(const std::vector<T> &data, const Shape &shape)
    {
        // 验证数据大小与形状是否匹配
        size_t expected_elements = shape.elements();
        if (data.size() != expected_elements)
        {
            throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                        ") does not match shape elements (" + std::to_string(expected_elements) + ")");
        }

        create_tensor_from_data(data.data(), data.size(), shape);
    }

    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape)
    {
        // 对于initializer_list，我们需要创建临时vector，因为迭代器不能直接转换为void*
        std::vector<T> data_vec(data);
        create_tensor_from_data(data_vec.data(), data_vec.size(), shape);
    }

    template <typename T>
    Tensor(T scalar, const Shape &shape)
    {
        create_tensor_from_scalar(scalar, shape);
    }

    // 标量构造函数（指定数据类型）
    template <typename T>
    Tensor(T scalar, const Shape &shape, DataType dtype)
    {
        create_tensor_from_scalar_with_dtype(&scalar, shape, dtype);
    }

    // === 支持DataType的构造函数（不给默认值）===
    template <typename T>
    Tensor(std::initializer_list<T> data, const Shape &shape, DataType dtype)
    {
        // 验证数据大小与形状是否匹配
        size_t expected_elements = shape.elements();
        if (data.size() != expected_elements)
        {
            throw std::invalid_argument("Data size (" + std::to_string(data.size()) +
                                        ") does not match shape elements (" + std::to_string(expected_elements) + ")");
        }

        // 创建临时vector并转换到指定类型
        std::vector<T> data_vec(data);
        create_tensor_from_data_with_dtype(data_vec.data(), data_vec.size(), shape, dtype);
    }

    // === 显式类型构造函数（不给默认值）===
    Tensor(const void *data, const Shape &shape, DataType dtype);

    // 拷贝构造函数 - 浅拷贝，共享实现
    Tensor(const Tensor &other);

    // 移动构造函数 - 转移所有权
    Tensor(Tensor &&other) noexcept;

    // 赋值运算符
    Tensor &operator=(const Tensor &other);
    Tensor &operator=(Tensor &&other) noexcept;

    // 析构函数
    ~Tensor() = default;

    // === 显式类型构造函数（不给默认值）===
    static Tensor from_blob(void *data, const Shape &shape, DataType dtype);

    // === 工厂方法（给默认值）===
    static Tensor zeros(const Shape &shape, DataType dtype = DataType::kFloat32);
    static Tensor ones(const Shape &shape, DataType dtype = DataType::kFloat32);
    static Tensor randn(const Shape &shape, DataType dtype = DataType::kFloat32);
    static Tensor full(
        const Shape &shape,
        double value,
        DataType dtype = DataType::kFloat32);  // TODO: 已经有了Tensor(T scalar, const Shape &shape)，此方法可以删除

    // === 形状和维度 ===
    Shape shape() const;
    size_t ndim() const;
    size_t elements() const;

    // === 数据访问：类型安全 ===
    template <typename T>
    T item() const;

    template <typename T>
    T *data_ptr();

    template <typename T>
    std::vector<T> to_vector() const;

    // === 类型查询和转换 ===
    DataType dtype() const;
    Tensor to(DataType target_type) const;

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

    // === 用于自动类型推断的方法 ===
    template <typename T>
    void create_tensor_from_scalar(T data, const Shape &shape);

    template <typename T>
    void create_tensor_from_data(const T *data, size_t count, const Shape &shape);

    // === 用于显式类型指定的方法 ===
    /**
     * @brief 从标量数据创建张量（显式指定类型）
     * @details 用于工厂方法如full()，标量数据是单个值，用void*传递，类型信息由dtype参数提供
     * @param data 标量数据的指针
     * @param shape 张量形状
     * @param dtype 目标数据类型
     */
    void create_tensor_from_scalar_with_dtype(const void *data, const Shape &shape, DataType dtype);

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