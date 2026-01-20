#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/core/tensor_options.h"
#include "origin/utils/exception.h"
#include <algorithm>
#include <cmath>
#include <memory>

// 前向声明 cpu::ones
namespace origin {
namespace cpu {
std::unique_ptr<origin::OriginMat> ones(const Shape &shape, const TensorOptions &options);
}  // namespace cpu
}  // namespace origin

namespace origin
{
namespace cpu
{

/**
 * @brief CPU yolo_detect_forward：YOLO Detect 前向传播（单个 stage）
 */
std::unique_ptr<Mat> yolo_detect_forward(const OriginMat &input,
                                         const OriginMat &conv_weight,
                                         const OriginMat *conv_bias,
                                         const OriginMat &grid,
                                         const OriginMat &anchor_grid,
                                         float stride,
                                         int32_t num_anchors,
                                         int32_t num_classes)
{
    // 检查输入形状
    auto input_shape = input.shape();
    if (unlikely(input_shape.size() != 4))
    {
        THROW_INVALID_ARG("yolo_detect_forward: input must be 4D (N, C, H, W), but got shape {}", input_shape.to_string());
    }

    size_t batch_size    = input_shape[0];
    size_t H             = input_shape[2];
    size_t W             = input_shape[3];
    int32_t classes_info = num_classes + 5;  // 4(bbox) + 1(objectness) + num_classes
    size_t num_boxes     = H * W * num_anchors;

    // 步骤1：卷积
    auto conv_output = input.conv2d(conv_weight, conv_bias, std::make_pair(1, 1), std::make_pair(0, 0));
    const origin::OriginMat &conv_output_mat = static_cast<const origin::OriginMat &>(*conv_output);

    // 步骤2：reshape + permute + reshape
    // reshape 为 (N, num_anchors, classes_info, H, W)
    Shape reshape_shape{batch_size, static_cast<size_t>(num_anchors), static_cast<size_t>(classes_info), H, W};
    auto reshaped = conv_output_mat.reshape(reshape_shape);
    const origin::OriginMat &reshaped_mat = static_cast<const origin::OriginMat &>(*reshaped);

    // permute 将 (N, num_anchors, classes_info, H, W) 转换为 (N, num_anchors, H, W, classes_info)
    auto permuted = reshaped_mat.permute({0, 1, 3, 4, 2});
    const origin::OriginMat &permuted_mat = static_cast<const origin::OriginMat &>(*permuted);

    // reshape 为 (N, num_boxes, classes_info)
    Shape target_shape{batch_size, num_boxes, static_cast<size_t>(classes_info)};
    auto stage_tensor = permuted_mat.reshape(target_shape);
    const origin::OriginMat &stage_tensor_mat = static_cast<const origin::OriginMat &>(*stage_tensor);

    // 步骤3：sigmoid
    // sigmoid(x) = 1 / (1 + exp(-x))
    auto neg_stage = -stage_tensor_mat;  // operator-()
    auto exp_neg_stage = neg_stage->exp();  // exp()
    
    // 创建全1矩阵
    TensorOptions options(input.dtype());
    options.device(input.device());
    auto ones_mat = origin::cpu::ones(target_shape, options);
    
    auto one_plus_exp = *ones_mat + *exp_neg_stage;  // operator+()
    auto sigmoid_output = *ones_mat / *one_plus_exp;  // operator/()
    const origin::OriginMat &sigmoid_output_mat = static_cast<const origin::OriginMat &>(*sigmoid_output);

    // 步骤4：坐标变换
    // xy = (xy * 2 + grid) * stride
    // wh = (wh * 2)^2 * anchor_grid
    
    // 创建输出矩阵
    auto result = std::make_unique<origin::OriginMat>(target_shape, input.dtype(), input.device());
    
    // 获取数据指针
    const float *sigmoid_data = sigmoid_output_mat.data_ptr<float>();
    float *result_data = result->data_ptr<float>();
    
    // 获取 grid 和 anchor_grid 数据
    // 在 cpu 命名空间内，OriginMat 就是 origin::OriginMat（通过 cpu_ops.h 引入）
    const float *grid_data = grid.data_ptr<float>();
    const float *anchor_grid_data = anchor_grid.data_ptr<float>();
    
    auto grid_shape = grid.shape();
    auto anchor_grid_shape = anchor_grid.shape();
    
    // 计算 anchor_grid 的实际尺寸
    size_t anchor_H = (anchor_grid_shape.size() >= 3) ? anchor_grid_shape[2] : H;
    size_t anchor_W = (anchor_grid_shape.size() >= 4) ? anchor_grid_shape[3] : W;
    
    // 进行坐标变换
    for (size_t b = 0; b < batch_size; ++b)
    {
        for (size_t i = 0; i < num_boxes; ++i)
        {
            size_t base_idx = b * num_boxes * classes_info + i * classes_info;
            
            // 计算在 grid 和 anchor_grid 中的索引
            size_t box_idx     = i;
            size_t anchor_idx  = box_idx / (H * W);
            size_t spatial_idx = box_idx % (H * W);
            size_t h_idx       = spatial_idx / W;
            size_t w_idx       = spatial_idx % W;
            
            // 获取 grid 值（假设 grid 已经展平为 (num_anchors * H * W * 2)）
            size_t grid_base = anchor_idx * H * W * 2 + h_idx * W * 2 + w_idx * 2;
            float grid_x     = (grid_base < grid_shape.elements()) ? grid_data[grid_base] : 0.0f;
            float grid_y     = (grid_base + 1 < grid_shape.elements()) ? grid_data[grid_base + 1] : 0.0f;
            
            // 获取 anchor_grid 值
            size_t scale_h      = (anchor_H > 0 && H > anchor_H) ? H / anchor_H : 1;
            size_t scale_w      = (anchor_W > 0 && W > anchor_W) ? W / anchor_W : 1;
            size_t anchor_h_idx = h_idx / scale_h;
            size_t anchor_w_idx = w_idx / scale_w;
            
            // 确保索引在有效范围内
            anchor_h_idx = std::min(anchor_h_idx, anchor_H - 1);
            anchor_w_idx = std::min(anchor_w_idx, anchor_W - 1);
            
            // 计算 anchor_grid 的索引（行主序）
            size_t anchor_grid_base =
                anchor_idx * anchor_H * anchor_W * 2 + anchor_h_idx * anchor_W * 2 + anchor_w_idx * 2;
            float anchor_w =
                (anchor_grid_base < anchor_grid_shape.elements()) ? anchor_grid_data[anchor_grid_base] : 1.0f;
            float anchor_h =
                (anchor_grid_base + 1 < anchor_grid_shape.elements()) ? anchor_grid_data[anchor_grid_base + 1] : 1.0f;
            
            // 变换坐标
            // x = (sigmoid_x * 2 + grid_x) * stride
            // y = (sigmoid_y * 2 + grid_y) * stride
            result_data[base_idx + 0] = (sigmoid_data[base_idx + 0] * 2.0f + grid_x) * stride;
            result_data[base_idx + 1] = (sigmoid_data[base_idx + 1] * 2.0f + grid_y) * stride;
            
            // w = (sigmoid_w * 2)^2 * anchor_w
            // h = (sigmoid_h * 2)^2 * anchor_h
            float sigmoid_w = sigmoid_data[base_idx + 2];
            float sigmoid_h = sigmoid_data[base_idx + 3];
            result_data[base_idx + 2] = std::pow(sigmoid_w * 2.0f, 2.0f) * anchor_w;
            result_data[base_idx + 3] = std::pow(sigmoid_h * 2.0f, 2.0f) * anchor_h;
            
            // 复制其他通道（objectness 和类别分数）
            for (size_t c = 4; c < static_cast<size_t>(classes_info); ++c)
            {
                result_data[base_idx + c] = sigmoid_data[base_idx + c];
            }
        }
    }
    
    return result;
}

}  // namespace cpu
}  // namespace origin
