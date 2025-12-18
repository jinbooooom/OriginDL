#ifndef __ORIGIN_DL_ORIGIN_MAT_H__
#define __ORIGIN_DL_ORIGIN_MAT_H__

#include <memory>
#include <vector>
#include "../../core/tensor_options.h"
#include "../basic_types.h"
#include "../mat.h"
#include "../shape.h"
#include "storage.h"

namespace origin
{

/**
 * @brief OriginMat后端的矩阵实现
 *
 * 这是OriginDL的自定义矩阵计算后端，使用Storage进行内存管理，
 * 支持CPU/GPU计算
 */
class OriginMat : public Mat
{
protected:
    std::shared_ptr<Storage> storage_;
    Shape shape_;
    DataType dtype_;
    std::vector<size_t> strides_;

private:
    // Helper to calculate strides
    std::vector<size_t> calculate_strides(const Shape &shape, DataType dtype);

    // Helper to get data type size
    size_t get_dtype_size(DataType dtype) const;

    // Helper to validate shape
    void validate_shape(const Shape &shape);

    // Helper to compute strides
    std::vector<size_t> compute_strides(const Shape &shape);

public:
    /**
     * @brief 默认构造函数
     */
    OriginMat() = default;

    /**
     * @brief 核心构造函数
     * @param storage 存储对象
     * @param shape 张量形状
     * @param dtype 数据类型
     */
    OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype);

    /**
     * @brief 视图构造函数（用于创建视图，共享Storage）
     * @param storage 存储对象
     * @param shape 张量形状
     * @param strides 步长信息
     * @param dtype 数据类型
     */
    OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, const std::vector<size_t> &strides, DataType dtype);

    // 为了向后兼容，保留一些构造函数
    OriginMat(const Shape &shape, DataType dtype);
    OriginMat(const Shape &shape, DataType dtype, Device device);

    // 两个核心工厂方法
    static std::unique_ptr<Mat> from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options);
    static std::unique_ptr<Mat> from_memory(const void *data,
                                            DataType user_dtype,
                                            const Shape &shape,
                                            const TensorOptions &options);

    // Mat interface implementations
    std::unique_ptr<Mat> clone() const override;
    std::unique_ptr<Mat> view(const Shape &shape) const override;
    bool is_contiguous() const override;
    std::unique_ptr<Mat> contiguous() const override;
    std::unique_ptr<Mat> reshape(const Shape &shape) const override;
    std::unique_ptr<Mat> transpose() const override;
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    void add_inplace(const Mat &other) override;
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    std::unique_ptr<Mat> operator-() const override;

    std::unique_ptr<Mat> square() const override;
    std::unique_ptr<Mat> pow(const Scalar &exponent) const override;
    std::unique_ptr<Mat> matmul(const Mat &other) const override;
    std::unique_ptr<Mat> sum(int axis) const override;
    std::unique_ptr<Mat> broadcast_to(const Shape &target_shape) const override;
    std::unique_ptr<Mat> sum_to(const Shape &target_shape) const override;
    bool can_broadcast_to(const Shape &target_shape) const;

    // 形状和维度
    Shape shape() const override;
    size_t elements() const override;

    // 数据访问
    std::vector<data_t> to_vector() const override;
    template <typename T>
    std::vector<T> to_vector() const
    {
        std::vector<T> result(shape_.elements());
        const T *data = data_ptr<T>();
        for (size_t i = 0; i < shape_.elements(); ++i)
        {
            result[i] = data[i];
        }
        return result;
    }

    // 数学函数
    std::unique_ptr<Mat> exp() const override;
    std::unique_ptr<Mat> log() const override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;

    // 0维张量支持
    bool is_scalar() const override;
    Scalar scalar_value() const override;

    // 类型和设备
    DataType dtype() const override;
    std::unique_ptr<Mat> to(DataType target_type) const override;
    Device device() const override;
    std::unique_ptr<Mat> to_device(Device device) const override;

    // 数据访问
    // 1. void* data_ptr() override: 虚函数版本，覆盖基类 Mat::data_ptr()，供 TensorImpl 通过基类指针调用
    // 2. template <typename T> T *data_ptr(): 模板函数，供内部实现代码（如 cpu/ 和 cuda/ 目录下的文件）直接通过 OriginMat 对象调用，提供类型安全
    // 3. template <typename T> const T *data_ptr() const: const 成员函数版本的模板函数，用于 const 对象的只读访问（通过 const 修饰符区分）
    void *data_ptr() override
    {
        return storage_->data();
    }

    template <typename T>
    T *data_ptr()
    {
        return static_cast<T *>(storage_->data());
    }

    template <typename T>
    const T *data_ptr() const
    {
        return static_cast<const T *>(storage_->data());
    }

    // 访问storage（用于CUDA运算）
    std::shared_ptr<Storage> storage() const { return storage_; }

    // 调试
    void print(const std::string &desc = "") const override;
    int backend_type() const override;

    // 工厂方法
    static std::unique_ptr<Mat> randn(const Shape &shape, const TensorOptions &options = TensorOptions());
    static std::unique_ptr<Mat> zeros(const Shape &shape, const TensorOptions &options = TensorOptions());
    static std::unique_ptr<Mat> ones(const Shape &shape, const TensorOptions &options = TensorOptions());
    static std::unique_ptr<Mat> full(const Shape &shape, data_t value, const TensorOptions &options = TensorOptions());

private:
    // Helper methods for type conversion
    template <typename T>
    const T *get_other_data(const OriginMat &other) const
    {
        return static_cast<const T *>(other.storage_->data());
    }
};

}  // namespace origin

#endif  // __ORIGIN_DL_ORIGIN_MAT_H__