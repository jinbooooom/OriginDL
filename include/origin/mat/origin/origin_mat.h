#ifndef __ORIGIN_DL_ORIGIN_MAT_H__
#define __ORIGIN_DL_ORIGIN_MAT_H__

#include <memory>
#include <vector>
#include "../../core/tensor_options.h"
#include "../basic_types.h"
#include "../mat.h"
#include "../shape.h"
#include "memory/storage.h"

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
    /**
     * @brief permute：按照指定顺序重新排列张量的维度
     * @param dims 新的维度顺序，例如 {0, 2, 3, 1} 表示将维度 0,1,2,3 重新排列为 0,2,3,1
     * @return 重排后的矩阵
     */
    std::unique_ptr<Mat> permute(const std::vector<int> &dims) const;
    std::unique_ptr<Mat> operator+(const Mat &other) const override;
    void add_inplace(const Mat &other) override;
    Mat &operator+=(const Mat &other) override;
    std::unique_ptr<Mat> operator-(const Mat &other) const override;
    void sub_inplace(const Mat &other) override;
    Mat &operator-=(const Mat &other) override;
    std::unique_ptr<Mat> operator*(const Mat &other) const override;
    void mul_inplace(const Mat &other) override;
    Mat &operator*=(const Mat &other) override;
    std::unique_ptr<Mat> operator/(const Mat &other) const override;
    void div_inplace(const Mat &other) override;
    Mat &operator/=(const Mat &other) override;
    std::unique_ptr<Mat> operator-() const override;

    // 比较运算符
    std::unique_ptr<Mat> operator==(const Mat &threshold) const override;
    std::unique_ptr<Mat> operator!=(const Mat &threshold) const override;
    std::unique_ptr<Mat> operator<(const Mat &threshold) const override;
    std::unique_ptr<Mat> operator<=(const Mat &threshold) const override;
    std::unique_ptr<Mat> operator>(const Mat &threshold) const override;
    std::unique_ptr<Mat> operator>=(const Mat &threshold) const override;

    std::unique_ptr<Mat> square() const override;
    std::unique_ptr<Mat> pow(const Scalar &exponent) const override;
    std::unique_ptr<Mat> matmul(const Mat &other) const override;
    std::unique_ptr<Mat> sum(int axis = -1, bool keepdim = false) const override;
    std::unique_ptr<Mat> max(int axis = -1) const override;
    std::unique_ptr<Mat> broadcast_to(const Shape &target_shape) const override;
    std::unique_ptr<Mat> sum_to(const Shape &target_shape) const override;
    bool can_broadcast_to(const Shape &target_shape) const;

    // === 卷积相关操作（Mat 接口实现）===
    /**
     * @brief im2col：将图像转换为列矩阵
     * @param kernel_size 卷积核大小
     * @param stride 步长
     * @param pad 填充
     * @param to_matrix 是否转换为矩阵形式
     * @return 列矩阵
     */
    std::unique_ptr<Mat> im2col(std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad,
                                bool to_matrix = true) const override;

    /**
     * @brief col2im：将列矩阵转换回图像形状
     * @param input_shape 原始输入形状 (N, C, H, W)
     * @param kernel_size 卷积核大小
     * @param stride 步长
     * @param pad 填充
     * @param to_matrix 是否从矩阵形式转换
     * @return 图像矩阵
     */
    std::unique_ptr<Mat> col2im(const Shape &input_shape,
                                std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad,
                                bool to_matrix = true) const override;

    /**
     * @brief conv2d：完整的二维卷积操作
     * @param W 卷积核 (OC, C, KH, KW)
     * @param b 偏置 (OC,)，可选，如果为 nullptr 则不添加偏置
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 输出张量 (N, OC, OH, OW)
     */
    std::unique_ptr<Mat> conv2d(const Mat &W,
                                const Mat *b,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad) const override;

    /**
     * @brief conv2d_backward：卷积反向传播
     * @param gy 输出梯度 (N, OC, OH, OW)
     * @param x 输入张量 (N, C, H, W)
     * @param W 卷积核 (OC, C, KH, KW)
     * @param b 偏置 (OC,)，可选
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 梯度向量：{gx, gW, [gb]}
     */
    std::vector<std::unique_ptr<Mat>> conv2d_backward(const Mat &gy,
                                                      const Mat &x,
                                                      const Mat &W,
                                                      const Mat *b,
                                                      std::pair<int, int> stride,
                                                      std::pair<int, int> pad) const override;

    // === 池化相关操作（Mat 接口实现）===
    /**
     * @brief avg_pool2d：平均池化操作
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 输出张量 (N, C, OH, OW)
     */
    std::unique_ptr<Mat> avg_pool2d(std::pair<int, int> kernel_size,
                                    std::pair<int, int> stride,
                                    std::pair<int, int> pad) const override;

    /**
     * @brief avg_pool2d_backward：平均池化反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 输入梯度 (N, C, H, W)
     */
    std::unique_ptr<Mat> avg_pool2d_backward(const Mat &gy,
                                             std::pair<int, int> kernel_size,
                                             std::pair<int, int> stride,
                                             std::pair<int, int> pad) const override;

    /**
     * @brief adaptive_avg_pool2d：自适应平均池化操作
     * @param output_size 输出尺寸 (OH, OW)
     * @return 输出张量 (N, C, OH, OW)
     */
    std::unique_ptr<Mat> adaptive_avg_pool2d(std::pair<int, int> output_size) const override;

    /**
     * @brief adaptive_avg_pool2d_backward：自适应平均池化反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param output_size 输出尺寸 (OH, OW)
     * @return 输入梯度 (N, C, H, W)
     */
    std::unique_ptr<Mat> adaptive_avg_pool2d_backward(const Mat &gy, std::pair<int, int> output_size) const override;

    /**
     * @brief max_pool2d：最大池化操作
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @param indices 输出参数：保存每个最大值在窗口内的索引
     * @return 输出张量 (N, C, OH, OW)
     */
    std::unique_ptr<Mat> max_pool2d(std::pair<int, int> kernel_size,
                                    std::pair<int, int> stride,
                                    std::pair<int, int> pad,
                                    std::vector<size_t> &indices) const override;

    /**
     * @brief max_pool2d_backward：最大池化反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @param indices 前向传播时保存的索引
     * @return 输入梯度 (N, C, H, W)
     */
    std::unique_ptr<Mat> max_pool2d_backward(const Mat &gy,
                                             std::pair<int, int> kernel_size,
                                             std::pair<int, int> stride,
                                             std::pair<int, int> pad,
                                             const std::vector<size_t> &indices) const override;

    // === 归一化相关操作（Mat 接口实现）===
    /**
     * @brief batch_norm_forward：BatchNorm 前向传播（返回所有中间结果）
     * @param gamma 缩放参数 (weight)，形状为 (C,)
     * @param beta 偏移参数 (bias)，形状为 (C,)
     * @param running_mean 运行均值，形状为 (C,)
     * @param running_var 运行方差，形状为 (C,)
     * @param training 是否为训练模式
     * @param eps 数值稳定性参数
     * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W)
     * @return BatchNormResult 包含输出和中间结果
     */
    BatchNormResult batch_norm_forward(const Mat &gamma,
                                       const Mat &beta,
                                       const Mat &running_mean,
                                       const Mat &running_var,
                                       bool training,
                                       float eps,
                                       int num_dims) const override;

    /**
     * @brief batch_norm：BatchNorm 前向传播（只返回输出）
     * @param gamma 缩放参数 (weight)，形状为 (C,)
     * @param beta 偏移参数 (bias)，形状为 (C,)
     * @param running_mean 运行均值，形状为 (C,)
     * @param running_var 运行方差，形状为 (C,)
     * @param training 是否为训练模式
     * @param eps 数值稳定性参数
     * @param momentum 动量参数
     * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W)
     * @return 输出张量，形状与输入相同
     */
    std::unique_ptr<Mat> batch_norm(const Mat &gamma,
                                    const Mat &beta,
                                    const Mat &running_mean,
                                    const Mat &running_var,
                                    bool training,
                                    float eps,
                                    float momentum,
                                    int num_dims) const override;

    /**
     * @brief batch_norm_backward：BatchNorm 反向传播
     * @param gy 输出梯度，形状与输入 x 相同
     * @param gamma 缩放参数 (weight)，形状为 (C,)
     * @param saved_mean 前向传播时保存的均值，形状为 (C,)
     * @param saved_var 前向传播时保存的方差，形状为 (C,)
     * @param saved_x_norm 前向传播时保存的归一化结果，形状与输入 x 相同
     * @param eps 数值稳定性参数
     * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W)
     * @return 梯度向量：{gx, dgamma, dbeta}
     */
    std::vector<std::unique_ptr<Mat>> batch_norm_backward(const Mat &gy,
                                                          const Mat &gamma,
                                                          const Mat &saved_mean,
                                                          const Mat &saved_var,
                                                          const Mat &saved_x_norm,
                                                          float eps,
                                                          int num_dims) const override;

    // === 其他操作（Mat 接口实现）===
    /**
     * @brief gather：根据索引从矩阵中提取值
     * @param indices 索引向量 (N,)，每个元素在 [0, C) 范围内
     * @return 提取的值 (N,)
     */
    std::unique_ptr<Mat> gather(const Mat &indices) const override;

    /**
     * @brief one_hot：将索引转换为 one-hot 编码
     * @param indices 索引向量 (N,)，每个元素在 [0, num_classes) 范围内
     * @param num_classes 类别数量
     * @return one-hot 编码矩阵 (N, num_classes)
     */
    std::unique_ptr<Mat> one_hot(const Mat &indices, int num_classes) const override;

    /**
     * @brief yolo_detect_forward：YOLO Detect 前向传播（单个 stage）
     * @param conv_weight 卷积权重 (OC, C, 1, 1)，其中 OC = num_anchors * (num_classes + 5)
     * @param conv_bias 卷积偏置 (OC,)，可选，如果为 nullptr 则不添加偏置
     * @param grid grid 坐标 (1, num_anchors, H, W, 2) 或展平后的形状
     * @param anchor_grid anchor grid 坐标 (1, num_anchors, anchor_H, anchor_W, 2) 或展平后的形状
     * @param stride stride 值
     * @param num_anchors anchor 数量
     * @param num_classes 类别数量
     * @return 输出张量 (N, num_boxes, classes_info)，其中 num_boxes = H * W * num_anchors, classes_info = num_classes +
     * 5
     */
    std::unique_ptr<Mat> yolo_detect_forward(const Mat &conv_weight,
                                             const Mat *conv_bias,
                                             const Mat &grid,
                                             const Mat &anchor_grid,
                                             float stride,
                                             int32_t num_anchors,
                                             int32_t num_classes) const override;

    // === Dropout 相关操作（Mat 接口实现）===
    /**
     * @brief dropout：Dropout 前向传播
     * @param p dropout 概率
     * @param training 是否为训练模式
     * @param mask 输出参数：保存 dropout mask（如果为 nullptr，则不保存 mask）
     * @return 输出张量
     */
    std::unique_ptr<Mat> dropout(float p, bool training, Mat *mask) const override;

    /**
     * @brief dropout_backward：Dropout 反向传播
     * @param gy 输出梯度
     * @param mask dropout mask
     * @return 输入梯度
     */
    std::unique_ptr<Mat> dropout_backward(const Mat &gy, const Mat &mask) const override;

    // === Upsample 相关操作（Mat 接口实现）===
    /**
     * @brief upsample：上采样操作
     * @param output_shape 输出形状 (N, C, OH, OW)
     * @param scale_h 高度缩放因子
     * @param scale_w 宽度缩放因子
     * @return 输出张量 (N, C, OH, OW)
     */
    std::unique_ptr<Mat> upsample(const Shape &output_shape, int scale_h, int scale_w) const override;

    /**
     * @brief upsample_backward：上采样反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param x_shape 输入形状 (N, C, H, W)
     * @param scale_h 高度缩放因子
     * @param scale_w 宽度缩放因子
     * @return 输入梯度 (N, C, H, W)
     */
    std::unique_ptr<Mat> upsample_backward(const Mat &gy,
                                           const Shape &x_shape,
                                           int scale_h,
                                           int scale_w) const override;

    // === Cat 和 Split 相关操作（Mat 接口实现）===
    /**
     * @brief cat：在指定维度上拼接多个矩阵
     * @param others 其他输入矩阵列表（所有矩阵必须具有相同的后端类型）
     * @param dim 拼接维度
     * @return 拼接后的矩阵
     */
    std::unique_ptr<Mat> cat(const std::vector<const Mat *> &others, int dim) const override;

    /**
     * @brief split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
     * @param output_shapes 输出形状列表
     * @param dim 分割维度
     * @return 分割后的矩阵列表
     */
    std::vector<std::unique_ptr<Mat>> split(const std::vector<Shape> &output_shapes, int dim) const override;

    // 形状和维度
    Shape shape() const override;
    const std::vector<size_t> &strides() const { return strides_; }
    size_t elements() const override;

    // 数据访问
    std::vector<float> to_vector() const override;
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
    void exp_inplace() override;
    std::unique_ptr<Mat> relu() const override;
    void relu_inplace() override;
    std::unique_ptr<Mat> log() const override;
    void log_inplace() override;
    std::unique_ptr<Mat> sin() const override;
    std::unique_ptr<Mat> cos() const override;
    std::unique_ptr<Mat> sqrt() const override;
    void sqrt_inplace() override;
    void square_inplace() override;
    void pow_inplace(const Scalar &exponent) override;
    void neg_inplace() override;

    // === 索引和选择操作 ===
    /**
     * @brief 根据多维索引读取单个元素
     * @param indices 多维索引，例如 {i, j, k} 表示访问 tensor[i][j][k]
     * @return 索引位置的值
     */
    Scalar index(std::initializer_list<size_t> indices) const;

    /**
     * @brief 根据多维索引写入单个元素
     * @param indices 多维索引，例如 {i, j, k} 表示访问 tensor[i][j][k]
     * @param value 要写入的标量值，与tensor相同的数据类型
     */
    void index_put(std::initializer_list<size_t> indices, const Scalar &value);

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
    // 2. template <typename T> T *data_ptr(): 模板函数，供内部实现代码（如 cpu/ 和 cuda/ 目录下的文件）直接通过
    // OriginMat 对象调用，提供类型安全
    // 3. template <typename T> const T *data_ptr() const: const 成员函数版本的模板函数，用于 const 对象的只读访问（通过
    // const 修饰符区分）
    void *data_ptr() override { return storage_->data(); }

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

    // 相等运算符（判断是否是同一块内存，比较 storage_）
    bool operator==(const OriginMat &other) const { return storage_ == other.storage_; }

    bool operator!=(const OriginMat &other) const { return !(*this == other); }

    // 调试
    void print(const std::string &desc = "") const override;
    int backend_type() const override;

    // 工厂方法
    static std::unique_ptr<Mat> randn(const Shape &shape, const TensorOptions &options = TensorOptions());
    static std::unique_ptr<Mat> zeros(const Shape &shape, const TensorOptions &options = TensorOptions());
    static std::unique_ptr<Mat> ones(const Shape &shape, const TensorOptions &options = TensorOptions());
    static std::unique_ptr<Mat> full(const Shape &shape, float value, const TensorOptions &options = TensorOptions());

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