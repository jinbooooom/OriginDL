#ifndef __ORIGIN_DL_MAT_H__
#define __ORIGIN_DL_MAT_H__

#include <memory>
#include <vector>
#include "basic_types.h"
#include "scalar.h"
#include "shape.h"

namespace origin
{

// 前向声明
class Shape;

/**
 * @brief 矩阵计算抽象接口
 * @details 定义了矩阵计算的基本操作，支持多种后端实现
 */
class Mat
{
public:
    virtual ~Mat() = default;

    /**
     * @brief 克隆矩阵（深拷贝，创建独立的数据副本）
     * @return 矩阵的副本
     */
    virtual std::unique_ptr<Mat> clone() const = 0;

    /**
     * @brief 创建视图（浅拷贝，共享底层存储，只改变形状）
     * @param shape 新的形状
     * @return 视图矩阵，共享底层存储
     * @note view() 要求元素总数必须匹配，且张量必须是连续的
     */
    virtual std::unique_ptr<Mat> view(const Shape &shape) const = 0;

    /**
     * @brief 检查矩阵是否在内存中连续存储
     * @return 如果矩阵是连续的返回true，否则返回false
     */
    virtual bool is_contiguous() const = 0;

    /**
     * @brief 创建连续存储的矩阵副本（如果已经是连续的则返回视图）
     * @return 连续存储的矩阵
     */
    virtual std::unique_ptr<Mat> contiguous() const = 0;

    /**
     * @brief 重塑矩阵形状
     * @param shape 新的形状
     * @return 重塑后的矩阵
     * @note 如果矩阵是连续的，会使用view()创建视图；否则会创建新存储并复制数据
     */
    virtual std::unique_ptr<Mat> reshape(const Shape &shape) const = 0;

    /**
     * @brief 转置矩阵
     * @return 转置后的矩阵
     */
    virtual std::unique_ptr<Mat> transpose() const = 0;

    /**
     * @brief 矩阵加法
     * @param other 另一个矩阵
     * @return 加法结果
     */
    virtual std::unique_ptr<Mat> operator+(const Mat &other) const = 0;

    /**
     * @brief 原地矩阵加法（累加到当前矩阵）
     * @param other 另一个矩阵
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void add_inplace(const Mat &other) = 0;

    /**
     * @brief 加法赋值运算符（累加到当前矩阵）
     * @param other 另一个矩阵
     * @return 当前矩阵的引用，支持链式调用
     */
    virtual Mat &operator+=(const Mat &other) = 0;

    /**
     * @brief 矩阵减法
     * @param other 另一个矩阵
     * @return 减法结果
     */
    virtual std::unique_ptr<Mat> operator-(const Mat &other) const = 0;

    /**
     * @brief 原地矩阵减法（从当前矩阵减去另一个矩阵）
     * @param other 另一个矩阵
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void sub_inplace(const Mat &other) = 0;

    /**
     * @brief 减法赋值运算符（从当前矩阵减去另一个矩阵）
     * @param other 另一个矩阵
     * @return 当前矩阵的引用，支持链式调用
     */
    virtual Mat &operator-=(const Mat &other) = 0;

    /**
     * @brief 元素级乘法
     * @param other 另一个矩阵
     * @return 乘法结果
     */
    virtual std::unique_ptr<Mat> operator*(const Mat &other) const = 0;

    /**
     * @brief 原地矩阵乘法（将当前矩阵与另一个矩阵相乘）
     * @param other 另一个矩阵
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void mul_inplace(const Mat &other) = 0;

    /**
     * @brief 乘法赋值运算符（将当前矩阵与另一个矩阵相乘）
     * @param other 另一个矩阵
     * @return 当前矩阵的引用，支持链式调用
     */
    virtual Mat &operator*=(const Mat &other) = 0;

    /**
     * @brief 矩阵乘法（真正的矩阵乘法）
     * @param other 另一个矩阵
     * @return 矩阵乘法结果
     */
    virtual std::unique_ptr<Mat> matmul(const Mat &other) const = 0;

    /**
     * @brief 矩阵除法
     * @param other 另一个矩阵
     * @return 除法结果
     */
    virtual std::unique_ptr<Mat> operator/(const Mat &other) const = 0;

    /**
     * @brief 原地矩阵除法（将当前矩阵除以另一个矩阵）
     * @param other 另一个矩阵
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void div_inplace(const Mat &other) = 0;

    /**
     * @brief 除法赋值运算符（将当前矩阵除以另一个矩阵）
     * @param other 另一个矩阵
     * @return 当前矩阵的引用，支持链式调用
     */
    virtual Mat &operator/=(const Mat &other) = 0;

    // === 比较运算符 ===
    /**
     * @brief 等于运算符
     * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
     * @return 比较结果mask，与输入相同类型和形状
     */
    virtual std::unique_ptr<Mat> operator==(const Mat &threshold) const = 0;

    /**
     * @brief 不等于运算符
     * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
     * @return 比较结果mask，与输入相同类型和形状
     */
    virtual std::unique_ptr<Mat> operator!=(const Mat &threshold) const = 0;

    /**
     * @brief 小于运算符
     * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
     * @return 比较结果mask，与输入相同类型和形状
     */
    virtual std::unique_ptr<Mat> operator<(const Mat &threshold) const = 0;

    /**
     * @brief 小于等于运算符
     * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
     * @return 比较结果mask，与输入相同类型和形状
     */
    virtual std::unique_ptr<Mat> operator<=(const Mat &threshold) const = 0;

    /**
     * @brief 大于运算符
     * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
     * @return 比较结果mask，与输入相同类型和形状
     */
    virtual std::unique_ptr<Mat> operator>(const Mat &threshold) const = 0;

    /**
     * @brief 大于等于运算符
     * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
     * @return 比较结果mask，与输入相同类型和形状
     */
    virtual std::unique_ptr<Mat> operator>=(const Mat &threshold) const = 0;



    /**
     * @brief 一元负号运算符
     * @return 负值结果
     */
    virtual std::unique_ptr<Mat> operator-() const = 0;

    /**
     * @brief 广播到指定形状
     * @param shape 目标形状
     * @return 广播后的矩阵
     */
    virtual std::unique_ptr<Mat> broadcast_to(const Shape &shape) const = 0;

    /**
     * @brief 求和到指定形状
     * @param shape 目标形状
     * @return 求和后的矩阵
     */
    virtual std::unique_ptr<Mat> sum_to(const Shape &shape) const = 0;

    /**
     * @brief 沿指定轴求和
     * @param axis 轴索引，-1表示所有元素
     * @param keepdim 是否保持维度，默认为false
     * @return 求和结果
     */
    virtual std::unique_ptr<Mat> sum(int axis = -1, bool keepdim = false) const = 0;

    /**
     * @brief 沿指定轴求最大值
     * @param axis 轴索引，-1表示所有元素
     * @return 最大值结果
     */
    virtual std::unique_ptr<Mat> max(int axis = -1) const = 0;

    /**
     * @brief 获取矩阵形状
     * @return 矩阵形状
     */
    virtual Shape shape() const = 0;

    /**
     * @brief 获取矩阵元素数量
     * @return 元素数量
     */
    virtual size_t elements() const = 0;

    /**
     * @brief 获取标量值（仅适用于单元素矩阵）
     * @return 标量值
     */
    template <typename T>
    T scalar() const;

    /**
     * @brief 判断是否为0维张量（标量张量）
     * @return 如果是0维张量返回true，否则返回false
     */
    virtual bool is_scalar() const = 0;

    /**
     * @brief 获取标量值（仅适用于0维张量）
     * @return 标量值
     */
    virtual Scalar scalar_value() const = 0;

    /**
     * @brief 根据多维索引读取单个元素
     * @param indices 多维索引，例如 {i, j, k} 表示访问 tensor[i][j][k]
     * @return 索引位置的值
     */
    virtual Scalar index(std::initializer_list<size_t> indices) const = 0;

    /**
     * @brief 根据多维索引写入单个元素
     * @param indices 多维索引，例如 {i, j, k} 表示访问 tensor[i][j][k]
     * @param value 要写入的标量值，会自动转换为与tensor相同的数据类型
     */
    virtual void index_put(std::initializer_list<size_t> indices, const Scalar &value) = 0;

    /**
     * @brief 获取数据指针
     * @return 指向数据的void*指针
     */
    virtual void *data_ptr() = 0;

    template <typename T>
    std::vector<T> to_vector() const;

    /**
     * @brief 打印矩阵内容
     * @param desc 描述信息
     */
    virtual void print(const std::string &desc = "") const = 0;

    /**
     * @brief 转换为向量
     * @return 矩阵数据的向量表示
     */
    virtual std::vector<float> to_vector() const = 0;  // TODO：不再硬编码返回std::vector<float>

    // 数学函数
    /**
     * @brief 指数函数
     * @return 指数运算结果
     */
    virtual std::unique_ptr<Mat> exp() const = 0;

    /**
     * @brief 原地指数函数（修改当前矩阵）
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void exp_inplace() = 0;

    /**
     * @brief 自然对数运算（以 e 为底）
     *
     * 计算矩阵的自然对数，即 log(x) = ln(x)
     *
     * @return 自然对数运算结果
     */
    virtual std::unique_ptr<Mat> log() const = 0;

    /**
     * @brief 原地自然对数函数（修改当前矩阵）
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void log_inplace() = 0;

    /**
     * @brief 正弦函数
     * @return 正弦运算结果
     */
    virtual std::unique_ptr<Mat> sin() const = 0;

    /**
     * @brief 余弦函数
     * @return 余弦运算结果
     */
    virtual std::unique_ptr<Mat> cos() const = 0;

    /**
     * @brief 平方根函数
     * @return 平方根运算结果
     */
    virtual std::unique_ptr<Mat> sqrt() const = 0;

    /**
     * @brief 原地平方根函数（修改当前矩阵）
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void sqrt_inplace() = 0;

    /**
     * @brief 平方函数
     * @return 平方运算结果
     */
    virtual std::unique_ptr<Mat> square() const = 0;

    /**
     * @brief 原地平方函数（修改当前矩阵）
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void square_inplace() = 0;

    /**
     * @brief 幂函数
     * @param exponent 指数
     * @return 幂运算结果
     */
    virtual std::unique_ptr<Mat> pow(const Scalar &exponent) const = 0;

    /**
     * @brief 原地幂函数（修改当前矩阵）
     * @param exponent 指数
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void pow_inplace(const Scalar &exponent) = 0;

    /**
     * @brief ReLU 激活函数
     * @return ReLU 运算结果，y = max(0, x)
     */
    virtual std::unique_ptr<Mat> relu() const = 0;

    /**
     * @brief 原地ReLU激活函数（修改当前矩阵）
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void relu_inplace() = 0;

    /**
     * @brief 原地取负函数（修改当前矩阵）
     * @note 原地操作，修改当前矩阵的数据，不创建新对象
     */
    virtual void neg_inplace() = 0;

    /**
     * @brief 获取后端类型
     * @return 后端类型标识符
     */
    virtual int backend_type() const = 0;

    /**
     * @brief 获取数据类型
     * @return 数据类型枚举
     */
    virtual DataType dtype() const = 0;

    /**
     * @brief 类型转换
     * @param target_type 目标数据类型
     * @return 转换后的矩阵
     */
    virtual std::unique_ptr<Mat> to(DataType target_type) const = 0;

    /**
     * @brief 获取设备信息
     * @return 当前设备
     */
    virtual Device device() const = 0;

    /**
     * @brief 设备转换
     * @param device 目标设备
     * @return 转换后的矩阵
     */
    virtual std::unique_ptr<Mat> to_device(Device device) const = 0;

    // === 卷积相关操作 ===
    /**
     * @brief im2col：将图像转换为列矩阵
     * @param kernel_size 卷积核大小
     * @param stride 步长
     * @param pad 填充
     * @param to_matrix 是否转换为矩阵形式
     * @return 列矩阵
     */
    virtual std::unique_ptr<Mat> im2col(std::pair<int, int> kernel_size,
                                        std::pair<int, int> stride,
                                        std::pair<int, int> pad,
                                        bool to_matrix = true) const = 0;

    /**
     * @brief col2im：将列矩阵转换回图像形状
     * @param input_shape 原始输入形状 (N, C, H, W)
     * @param kernel_size 卷积核大小
     * @param stride 步长
     * @param pad 填充
     * @param to_matrix 是否从矩阵形式转换
     * @return 图像矩阵
     */
    virtual std::unique_ptr<Mat> col2im(const Shape &input_shape,
                                        std::pair<int, int> kernel_size,
                                        std::pair<int, int> stride,
                                        std::pair<int, int> pad,
                                        bool to_matrix = true) const = 0;

    /**
     * @brief conv2d：完整的二维卷积操作
     * @param W 卷积核 (OC, C, KH, KW)
     * @param b 偏置 (OC,)，可选，如果为 nullptr 则不添加偏置
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 输出张量 (N, OC, OH, OW)
     */
    virtual std::unique_ptr<Mat> conv2d(const Mat &W,
                                        const Mat *b,
                                        std::pair<int, int> stride,
                                        std::pair<int, int> pad) const = 0;

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
    virtual std::vector<std::unique_ptr<Mat>> conv2d_backward(const Mat &gy,
                                                              const Mat &x,
                                                              const Mat &W,
                                                              const Mat *b,
                                                              std::pair<int, int> stride,
                                                              std::pair<int, int> pad) const = 0;

    // === 池化相关操作 ===
    /**
     * @brief avg_pool2d：平均池化操作
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 输出张量 (N, C, OH, OW)
     */
    virtual std::unique_ptr<Mat> avg_pool2d(std::pair<int, int> kernel_size,
                                            std::pair<int, int> stride,
                                            std::pair<int, int> pad) const = 0;

    /**
     * @brief avg_pool2d_backward：平均池化反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @return 输入梯度 (N, C, H, W)
     */
    virtual std::unique_ptr<Mat> avg_pool2d_backward(const Mat &gy,
                                                     std::pair<int, int> kernel_size,
                                                     std::pair<int, int> stride,
                                                     std::pair<int, int> pad) const = 0;

    /**
     * @brief adaptive_avg_pool2d：自适应平均池化操作
     * @param output_size 输出尺寸 (OH, OW)
     * @return 输出张量 (N, C, OH, OW)
     */
    virtual std::unique_ptr<Mat> adaptive_avg_pool2d(std::pair<int, int> output_size) const = 0;

    /**
     * @brief adaptive_avg_pool2d_backward：自适应平均池化反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param output_size 输出尺寸 (OH, OW)
     * @return 输入梯度 (N, C, H, W)
     */
    virtual std::unique_ptr<Mat> adaptive_avg_pool2d_backward(const Mat &gy, std::pair<int, int> output_size) const = 0;

    /**
     * @brief max_pool2d：最大池化操作
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @param indices 输出参数：保存每个最大值在窗口内的索引
     * @return 输出张量 (N, C, OH, OW)
     */
    virtual std::unique_ptr<Mat> max_pool2d(std::pair<int, int> kernel_size,
                                            std::pair<int, int> stride,
                                            std::pair<int, int> pad,
                                            std::vector<size_t> &indices) const = 0;

    /**
     * @brief max_pool2d_backward：最大池化反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param kernel_size 池化核大小 (KH, KW)
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @param indices 前向传播时保存的索引
     * @return 输入梯度 (N, C, H, W)
     */
    virtual std::unique_ptr<Mat> max_pool2d_backward(const Mat &gy,
                                                     std::pair<int, int> kernel_size,
                                                     std::pair<int, int> stride,
                                                     std::pair<int, int> pad,
                                                     const std::vector<size_t> &indices) const = 0;

    // === BatchNorm 相关操作 ===
    /**
     * @brief BatchNorm 前向传播结果结构体
     */
    struct BatchNormResult
    {
        std::unique_ptr<Mat> y;       // 输出
        std::unique_ptr<Mat> mean;    // 当前 batch 的均值
        std::unique_ptr<Mat> var;     // 当前 batch 的方差
        std::unique_ptr<Mat> x_norm;  // 归一化后的 x
    };

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
    virtual BatchNormResult batch_norm_forward(const Mat &gamma,
                                               const Mat &beta,
                                               const Mat &running_mean,
                                               const Mat &running_var,
                                               bool training,
                                               float eps,
                                               int num_dims) const = 0;

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
    virtual std::unique_ptr<Mat> batch_norm(const Mat &gamma,
                                            const Mat &beta,
                                            const Mat &running_mean,
                                            const Mat &running_var,
                                            bool training,
                                            float eps,
                                            float momentum,
                                            int num_dims) const = 0;

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
    virtual std::vector<std::unique_ptr<Mat>> batch_norm_backward(const Mat &gy,
                                                                  const Mat &gamma,
                                                                  const Mat &saved_mean,
                                                                  const Mat &saved_var,
                                                                  const Mat &saved_x_norm,
                                                                  float eps,
                                                                  int num_dims) const = 0;

    // === 其他操作 ===
    /**
     * @brief gather：根据索引从矩阵中提取值
     * @param indices 索引向量 (N,)，每个元素在 [0, C) 范围内
     * @return 提取的值 (N,)
     */
    virtual std::unique_ptr<Mat> gather(const Mat &indices) const = 0;

    /**
     * @brief one_hot：将索引转换为 one-hot 编码
     * @param indices 索引向量 (N,)，每个元素在 [0, num_classes) 范围内
     * @param num_classes 类别数量
     * @return one-hot 编码矩阵 (N, num_classes)
     */
    virtual std::unique_ptr<Mat> one_hot(const Mat &indices, int num_classes) const = 0;

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
    virtual std::unique_ptr<Mat> yolo_detect_forward(const Mat &conv_weight,
                                                     const Mat *conv_bias,
                                                     const Mat &grid,
                                                     const Mat &anchor_grid,
                                                     float stride,
                                                     int32_t num_anchors,
                                                     int32_t num_classes) const = 0;

    // === Dropout 相关操作 ===
    /**
     * @brief dropout：Dropout 前向传播
     * @param p dropout 概率
     * @param training 是否为训练模式
     * @param mask 输出参数：保存 dropout mask（如果为 nullptr，则不保存 mask）
     * @return 输出张量
     */
    virtual std::unique_ptr<Mat> dropout(float p, bool training, Mat *mask) const = 0;

    /**
     * @brief dropout_backward：Dropout 反向传播
     * @param gy 输出梯度
     * @param mask dropout mask
     * @return 输入梯度
     */
    virtual std::unique_ptr<Mat> dropout_backward(const Mat &gy, const Mat &mask) const = 0;

    // === Upsample 相关操作 ===
    /**
     * @brief upsample：上采样操作
     * @param output_shape 输出形状 (N, C, OH, OW)
     * @param scale_h 高度缩放因子
     * @param scale_w 宽度缩放因子
     * @return 输出张量 (N, C, OH, OW)
     */
    virtual std::unique_ptr<Mat> upsample(const Shape &output_shape, int scale_h, int scale_w) const = 0;

    /**
     * @brief upsample_backward：上采样反向传播
     * @param gy 输出梯度 (N, C, OH, OW)
     * @param x_shape 输入形状 (N, C, H, W)
     * @param scale_h 高度缩放因子
     * @param scale_w 宽度缩放因子
     * @return 输入梯度 (N, C, H, W)
     */
    virtual std::unique_ptr<Mat> upsample_backward(const Mat &gy,
                                                   const Shape &x_shape,
                                                   int scale_h,
                                                   int scale_w) const = 0;

    // === Cat 和 Split 相关操作 ===
    /**
     * @brief cat：在指定维度上拼接多个矩阵
     * @param others 其他输入矩阵列表（所有矩阵必须具有相同的后端类型）
     * @param dim 拼接维度
     * @return 拼接后的矩阵
     */
    virtual std::unique_ptr<Mat> cat(const std::vector<const Mat *> &others, int dim) const = 0;

    /**
     * @brief split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
     * @param output_shapes 输出形状列表
     * @param dim 分割维度
     * @return 分割后的矩阵列表
     */
    virtual std::vector<std::unique_ptr<Mat>> split(const std::vector<Shape> &output_shapes, int dim) const = 0;
};

/**
 * @brief Mat工厂函数，用于创建Mat对象而不暴露具体后端类型
 * @param data 数据向量
 * @param shape 矩阵形状
 * @return Mat对象的智能指针
 */
std::unique_ptr<Mat> create_mat(const std::vector<float> &data, const Shape &shape);

/**
 * @brief Mat工厂函数，用于创建标量矩阵
 * @param value 标量值
 * @param shape 矩阵形状
 * @return Mat对象的智能指针
 */
std::unique_ptr<Mat> create_mat(float value, const Shape &shape);

template <typename T>
std::vector<T> Mat::to_vector() const
{
    // 调用现有的虚函数，然后转换类型
    auto float_vec = to_vector();
    std::vector<T> result;
    result.reserve(float_vec.size());
    for (const auto &val : float_vec)
    {
        result.push_back(static_cast<T>(val));
    }
    return result;
}

}  // namespace origin

#endif  // __ORIGIN_DL_MAT_H__
