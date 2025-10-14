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
 * 支持CPU计算，为未来的CUDA支持预留接口
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

    // Helper to get data type from template
    template <typename T>
    DataType get_data_type_from_template() const;

public:
    /**
     * @brief 默认构造函数
     */
    OriginMat() = default;

    /**
     * @brief 从Storage构造
     * @param storage 存储对象
     * @param shape 张量形状
     * @param dtype 数据类型
     */
    OriginMat(std::shared_ptr<Storage> storage, const Shape &shape, DataType dtype);
    OriginMat(const Shape &shape, DataType dtype);

    /**
     * @brief 通用构造函数：从不同数据类型创建
     * @param data 数据向量
     * @param shape 矩阵形状
     */
    template <typename T>
    OriginMat(const std::vector<T> &data, const Shape &shape);

    /**
     * @brief 标量构造函数
     * @param value 标量值
     * @param shape 矩阵形状
     */
    template <typename T>
    OriginMat(T value, const Shape &shape);

    /**
     * @brief 带TensorOptions的构造函数
     */
    template <typename T>
    OriginMat(const std::vector<T> &data, const Shape &shape, const TensorOptions &options);

    template <typename T>
    OriginMat(T value, const Shape &shape, const TensorOptions &options);

    // Mat interface implementations
    std::unique_ptr<Mat> clone() const override;
    std::unique_ptr<Mat> reshape(const Shape &shape) const override;
    std::unique_ptr<Mat> transpose() const override;
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    std::unique_ptr<Mat> operator+(data_t scalar) const override;
    std::unique_ptr<Mat> operator-(data_t scalar) const override;
    std::unique_ptr<Mat> operator*(data_t scalar) const override;
    std::unique_ptr<Mat> operator/(data_t scalar) const override;
    std::unique_ptr<Mat> add_scalar(data_t scalar) const override;
    std::unique_ptr<Mat> mul_scalar(data_t scalar) const override;
    std::unique_ptr<Mat> operator-() const override;
    std::unique_ptr<Mat> square() const override;
    std::unique_ptr<Mat> pow(data_t exponent) const override;
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

    // 统计函数
    data_t sum_all() const override;
    data_t max_all() const override;
    data_t min_all() const override;
    data_t mean_all() const override;

    // 类型和设备
    DataType dtype() const override;
    std::unique_ptr<Mat> to(DataType target_type) const override;
    Device device() const override;
    std::unique_ptr<Mat> to_device(Device device) const override;

    // 数据访问
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

    // 调试
    void print(const std::string &desc = "") const override;
    int backend_type() const override;
    
    // 访问storage_的公共方法
    std::shared_ptr<Storage> get_storage() const { return storage_; }

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