#ifndef __ORIGIN_DL_TORCH_MAT_H__
#define __ORIGIN_DL_TORCH_MAT_H__

#include <torch/torch.h>
#include <memory>
#include <vector>
#include "../../core/tensor_options.h"
#include "../basic_types.h"
#include "../mat.h"
#include "../shape.h"

namespace origin
{

/**
 * @brief TorchMat后端的矩阵实现
 *
 * 这是基于LibTorch的矩阵计算后端实现
 * 注意：当前很多操作尚未实现，会抛出不支持异常
 */
class TorchMat : public Mat
{
public:
    // 构造函数
    TorchMat() = default;
    explicit TorchMat(const torch::Tensor &tensor);
    explicit TorchMat(torch::Tensor &&tensor);

    // === 静态工厂方法 ===
    static std::unique_ptr<Mat> from_scalar(const Scalar &scalar, const Shape &shape, const TensorOptions &options);
    static std::unique_ptr<Mat> from_memory(const void *data,
                                            DataType user_dtype,
                                            const Shape &shape,
                                            const TensorOptions &options);
    static std::unique_ptr<Mat> randn(const Shape &shape, const TensorOptions &options = TensorOptions());

    // === Mat 接口实现 ===
    // 基本操作（需要实现，但这里先占位）
    std::unique_ptr<Mat> clone() const override;
    std::unique_ptr<Mat> view(const Shape &shape) const override;
    bool is_contiguous() const override;
    std::unique_ptr<Mat> contiguous() const override;
    std::unique_ptr<Mat> reshape(const Shape &shape) const override;
    std::unique_ptr<Mat> transpose() const override;

    // 二元运算
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    void add_inplace(const Mat &other) override;
    Mat &operator+=(const Mat &other) override;  // 返回 Mat&，是为了支持链式调用（y += x += z; // 先算 x+=z，再 y+=）
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    void sub_inplace(const Mat &other) override;
    Mat &operator-=(const Mat &other) override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    void mul_inplace(const Mat &other) override;
    Mat &operator*=(const Mat &other) override;
    std::unique_ptr<Mat> matmul(const Mat &other) const override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    void div_inplace(const Mat &other) override;
    Mat &operator/=(const Mat &other) override;
    std::unique_ptr<Mat> operator-() const override;
    std::unique_ptr<Mat> operator>(const Scalar &threshold) const override;

    // 广播和归约
    std::unique_ptr<Mat> broadcast_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum_to(const Shape &shape) const override;
    std::unique_ptr<Mat> sum(int axis = -1, bool keepdim = false) const override;
    std::unique_ptr<Mat> max(int axis = -1) const override;

    // 数学函数
    std::unique_ptr<Mat> exp() const override;
    void exp_inplace() override;
    std::unique_ptr<Mat> log() const override;
    void log_inplace() override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;
    void sqrt_inplace() override;
    std::unique_ptr<Mat> square() const override;
    void square_inplace() override;
    std::unique_ptr<Mat> pow(const Scalar &exponent) const override;
    void pow_inplace(const Scalar &exponent) override;
    std::unique_ptr<Mat> relu() const override;
    void relu_inplace() override;
    void neg_inplace() override;

    // 形状和属性
    Shape shape() const override;
    size_t elements() const override;
    bool is_scalar() const override;
    Scalar scalar_value() const override;
    Scalar index(std::initializer_list<size_t> indices) const override;
    void index_put(std::initializer_list<size_t> indices, const Scalar &value) override;
    void *data_ptr() override;
    void print(const std::string &desc = "") const override;
    std::vector<float> to_vector() const;

    // 类型和设备
    int backend_type() const override;
    DataType dtype() const override;
    Device device() const override;
    std::unique_ptr<Mat> to(DataType target_type) const override;
    std::unique_ptr<Mat> to_device(Device device) const override;

    // === 卷积相关操作（Mat 接口实现）===
    std::unique_ptr<Mat> im2col(std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad,
                                bool to_matrix = true) const override;

    std::unique_ptr<Mat> col2im(const Shape &input_shape,
                                std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad,
                                bool to_matrix = true) const override;

    std::unique_ptr<Mat> conv2d(const Mat &W,
                                const Mat *b,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad) const override;

    std::vector<std::unique_ptr<Mat>> conv2d_backward(const Mat &gy,
                                                      const Mat &x,
                                                      const Mat &W,
                                                      const Mat *b,
                                                      std::pair<int, int> stride,
                                                      std::pair<int, int> pad) const override;

    // === 池化相关操作（Mat 接口实现）===
    std::unique_ptr<Mat> avg_pool2d(std::pair<int, int> kernel_size,
                                    std::pair<int, int> stride,
                                    std::pair<int, int> pad) const override;

    std::unique_ptr<Mat> avg_pool2d_backward(const Mat &gy,
                                             std::pair<int, int> kernel_size,
                                             std::pair<int, int> stride,
                                             std::pair<int, int> pad) const override;

    std::unique_ptr<Mat> adaptive_avg_pool2d(std::pair<int, int> output_size) const override;

    std::unique_ptr<Mat> adaptive_avg_pool2d_backward(const Mat &gy, std::pair<int, int> output_size) const override;

    std::unique_ptr<Mat> max_pool2d(std::pair<int, int> kernel_size,
                                    std::pair<int, int> stride,
                                    std::pair<int, int> pad,
                                    std::vector<size_t> &indices) const override;

    std::unique_ptr<Mat> max_pool2d_backward(const Mat &gy,
                                             std::pair<int, int> kernel_size,
                                             std::pair<int, int> stride,
                                             std::pair<int, int> pad,
                                             const std::vector<size_t> &indices) const override;

    // === BatchNorm 相关操作（Mat 接口实现）===
    BatchNormResult batch_norm_forward(const Mat &gamma,
                                       const Mat &beta,
                                       const Mat &running_mean,
                                       const Mat &running_var,
                                       bool training,
                                       float eps,
                                       int num_dims) const override;

    std::unique_ptr<Mat> batch_norm(const Mat &gamma,
                                    const Mat &beta,
                                    const Mat &running_mean,
                                    const Mat &running_var,
                                    bool training,
                                    float eps,
                                    float momentum,
                                    int num_dims) const override;

    std::vector<std::unique_ptr<Mat>> batch_norm_backward(const Mat &gy,
                                                          const Mat &gamma,
                                                          const Mat &saved_mean,
                                                          const Mat &saved_var,
                                                          const Mat &saved_x_norm,
                                                          float eps,
                                                          int num_dims) const override;

    // === 其他操作（Mat 接口实现）===
    std::unique_ptr<Mat> gather(const Mat &indices) const override;

    std::unique_ptr<Mat> one_hot(const Mat &indices, int num_classes) const override;

    std::unique_ptr<Mat> yolo_detect_forward(const Mat &conv_weight,
                                             const Mat *conv_bias,
                                             const Mat &grid,
                                             const Mat &anchor_grid,
                                             float stride,
                                             int32_t num_anchors,
                                             int32_t num_classes) const override;

    // === Dropout 相关操作（Mat 接口实现）===
    std::unique_ptr<Mat> dropout(float p, bool training, Mat *mask) const override;

    std::unique_ptr<Mat> dropout_backward(const Mat &gy, const Mat &mask) const override;

    // === Upsample 相关操作（Mat 接口实现）===
    std::unique_ptr<Mat> upsample(const Shape &output_shape,
                                  int scale_h,
                                  int scale_w,
                                  const std::string &mode = "nearest") const override;

    std::unique_ptr<Mat> upsample_backward(const Mat &gy,
                                           const Shape &x_shape,
                                           int scale_h,
                                           int scale_w,
                                           const std::string &mode = "nearest") const override;

private:
    torch::Tensor tensor_;
};

}  // namespace origin

#endif  // __ORIGIN_DL_TORCH_MAT_H__
