#include "origin/operators/custom/yolo_detect.h"
#include <cmath>
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/operators/conv/conv2d.h"
#include "origin/operators/shape/cat.h"
#include "origin/utils/exception.h"
#include "origin/utils/log.h"

namespace origin
{
namespace functional
{

YoloDetect::YoloDetect(int32_t stages,
                       int32_t num_classes,
                       int32_t num_anchors,
                       std::vector<float> strides,
                       std::vector<Tensor> anchor_grids,
                       std::vector<Tensor> grids,
                       std::vector<Tensor> conv_weights,
                       std::vector<Tensor> conv_biases)
    : stages_(stages),
      num_classes_(num_classes),
      num_anchors_(num_anchors),
      strides_(std::move(strides)),
      anchor_grids_(std::move(anchor_grids)),
      grids_(std::move(grids)),
      conv_weights_(std::move(conv_weights)),
      conv_biases_(std::move(conv_biases))
{
    if (stages_ != 3)
    {
        THROW_RUNTIME_ERROR("YoloDetect only supports 3 stages, but got {}", stages_);
    }
    if (strides_.size() != static_cast<size_t>(stages_))
    {
        THROW_RUNTIME_ERROR("YoloDetect: strides size ({}) does not match stages ({})", strides_.size(), stages_);
    }
    if (anchor_grids_.size() != static_cast<size_t>(stages_))
    {
        THROW_RUNTIME_ERROR("YoloDetect: anchor_grids size ({}) does not match stages ({})", anchor_grids_.size(),
                            stages_);
    }
    if (grids_.size() != static_cast<size_t>(stages_))
    {
        THROW_RUNTIME_ERROR("YoloDetect: grids size ({}) does not match stages ({})", grids_.size(), stages_);
    }
    if (conv_weights_.size() != static_cast<size_t>(stages_))
    {
        THROW_RUNTIME_ERROR("YoloDetect: conv_weights size ({}) does not match stages ({})", conv_weights_.size(),
                            stages_);
    }
    if (conv_biases_.size() != static_cast<size_t>(stages_))
    {
        THROW_RUNTIME_ERROR("YoloDetect: conv_biases size ({}) does not match stages ({})", conv_biases_.size(),
                            stages_);
    }
}

std::vector<Tensor> YoloDetect::forward(const std::vector<Tensor> &xs)
{
    if (xs.size() != static_cast<size_t>(stages_))
    {
        THROW_RUNTIME_ERROR("YoloDetect forward: expected {} inputs (stages), but got {}", stages_, xs.size());
    }

    // 获取 batch size
    size_t batch_size    = xs[0].shape()[0];
    int32_t classes_info = num_classes_ + 5;  // 4(bbox) + 1(objectness) + num_classes

    // 存储每个阶段的输出
    std::vector<Tensor> stage_outputs;

    // 对每个阶段进行处理
    for (int32_t stage = 0; stage < stages_; ++stage)
    {
        const Tensor &input = xs[stage];
        auto input_shape    = input.shape();

        // 检查输入形状：应该是 (N, C, H, W)
        if (input_shape.size() != 4)
        {
            THROW_RUNTIME_ERROR("YoloDetect forward: input {} must be 4D (N, C, H, W), but got shape {}", stage,
                                input_shape.to_string());
        }

        size_t H         = input_shape[2];
        size_t W         = input_shape[3];
        size_t num_boxes = H * W * num_anchors_;

        // 步骤1：对输入特征图应用卷积
        // 使用 Conv2dOp 进行卷积操作
        auto conv_op = std::make_shared<functional::Conv2dOp>(std::make_pair(1, 1), std::make_pair(0, 0));
        std::vector<Tensor> conv_inputs = {input, conv_weights_[stage], conv_biases_[stage]};
        auto conv_outputs               = conv_op->forward(conv_inputs);
        Tensor conv_output              = conv_outputs[0];

        // conv_output 形状应该是 (N, OC, H, W)，其中 OC = num_anchors * (num_classes + 5)
        auto conv_shape = conv_output.shape();
        if (conv_shape[2] != H || conv_shape[3] != W)
        {
            THROW_RUNTIME_ERROR("YoloDetect forward: conv output shape mismatch at stage {}", stage);
        }

        // 步骤2：reshape 为 (N, num_anchors, classes_info, H, W)
        // 然后转置/重组为 (N, num_anchors * H * W, classes_info)
        Shape reshape_shape{batch_size, static_cast<size_t>(num_anchors_), static_cast<size_t>(classes_info), H, W};
        Tensor reshaped = conv_output.reshape(reshape_shape);

        // 重新排列为 (N, num_anchors * H * W, classes_info)
        // 需要将 (N, num_anchors, classes_info, H, W) 转换为 (N, num_anchors * H * W, classes_info)
        // 这需要手动重新排列数据
        Shape target_shape{batch_size, num_boxes, static_cast<size_t>(classes_info)};

        // 将数据复制到 CPU 进行重新排列
        auto reshaped_data = reshaped.to_vector<float>();
        std::vector<float> output_data(target_shape.elements());

        // 重新排列数据：从 (N, num_anchors, classes_info, H, W) 到 (N, num_anchors * H * W, classes_info)
        // 对于行主序数据布局，索引计算应该是：
        // input_idx = b * num_anchors * classes_info * H * W + na * classes_info * H * W + c * H * W + h * W + w
        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t na = 0; na < static_cast<size_t>(num_anchors_); ++na)
            {
                for (size_t h = 0; h < H; ++h)
                {
                    for (size_t w = 0; w < W; ++w)
                    {
                        size_t output_idx = b * num_boxes * classes_info + (na * H * W + h * W + w) * classes_info;

                        for (size_t c = 0; c < static_cast<size_t>(classes_info); ++c)
                        {
                            // 行主序索引：b * num_anchors * classes_info * H * W + na * classes_info * H * W + c * H *
                            // W + h * W + w
                            size_t input_idx = b * num_anchors_ * classes_info * H * W + na * classes_info * H * W +
                                               c * H * W + h * W + w;
                            output_data[output_idx + c] = reshaped_data[input_idx];
                        }
                    }
                }
            }
        }

        Tensor stage_tensor(output_data, target_shape, dtype(DataType::kFloat32).device(input.device()));

        // if (stage == 0) {
        //     size_t batch_idx = 0;
        //     size_t box_idx = 0;
        //     size_t base_idx = batch_idx * num_boxes * classes_info + box_idx * classes_info;
        //     logd("YoloDetect Stage {} Before Sigmoid: First box raw values (first 5): {}", stage, ...);
        // }

        // 步骤3：应用 sigmoid 到坐标部分（前4个通道：x, y, w, h）
        // 注意：这里只对前4个通道应用 sigmoid，objectness 和类别分数保持原样
        // 但实际上，根据 YOLOv5 的实现，前4个通道需要 sigmoid，第5个通道（objectness）也需要 sigmoid

        // 提取坐标部分 (N, num_boxes, 4) 和 objectness/类别部分 (N, num_boxes, classes_info - 4)
        // 由于 Tensor API 的限制，我们需要手动处理

        // 简化实现：对整个 tensor 应用 sigmoid（虽然只有前5个通道需要）
        // 更精确的实现需要分别处理不同通道
        auto sigmoid_op                    = std::make_shared<functional::Sigmoid>();
        std::vector<Tensor> sigmoid_inputs = {stage_tensor};
        auto sigmoid_outputs               = sigmoid_op->forward(sigmoid_inputs);
        Tensor sigmoid_output              = sigmoid_outputs[0];

        // if (stage == 0) {
        //     auto sigmoid_data = sigmoid_output.to_vector<float>();
        //     size_t batch_idx = 0;
        //     size_t box_idx = 0;
        //     size_t base_idx = batch_idx * num_boxes * classes_info + box_idx * classes_info;
        //     logd("YoloDetect Stage {} After Sigmoid: First box sigmoid values (first 5): {}", stage, ...);
        // }

        // 步骤4：进行坐标变换
        // xy = (xy * 2 + grid) * stride
        // wh = (wh * 2)^2 * anchor_grid

        // 将数据复制到 CPU 进行处理
        auto sigmoid_data = sigmoid_output.to_vector<float>();

        // 获取 grid 和 anchor_grid 数据
        auto grid_shape        = grids_[stage].shape();
        auto anchor_grid_shape = anchor_grids_[stage].shape();
        auto grid_data         = grids_[stage].to_vector<float>();
        auto anchor_grid_data  = anchor_grids_[stage].to_vector<float>();
        float stride           = strides_[stage];

        // 进行坐标变换
        for (size_t b = 0; b < batch_size; ++b)
        {
            for (size_t i = 0; i < num_boxes; ++i)
            {
                size_t base_idx = b * num_boxes * classes_info + i * classes_info;

                // 计算在 grid 和 anchor_grid 中的索引
                size_t box_idx     = i;                  // 在 num_boxes 中的索引
                size_t anchor_idx  = box_idx / (H * W);  // anchor 索引
                size_t spatial_idx = box_idx % (H * W);  // 空间位置索引
                size_t h_idx       = spatial_idx / W;
                size_t w_idx       = spatial_idx % W;

                // 获取 grid 值（grid 形状为 (1, num_anchors, H, W, 2)）
                // 简化：假设 grid 已经展平为 (num_anchors * H * W * 2)
                size_t grid_base = anchor_idx * H * W * 2 + h_idx * W * 2 + w_idx * 2;
                float grid_x     = (grid_base < grid_data.size()) ? grid_data[grid_base] : 0.0f;
                float grid_y     = (grid_base + 1 < grid_data.size()) ? grid_data[grid_base + 1] : 0.0f;

                // 获取 anchor_grid 值
                // anchor_grid 的形状是 (1, num_anchors, anchor_H, anchor_W, 2)
                // 其中 anchor_H 和 anchor_W 可能小于输入特征图的 H 和 W
                // 需要根据 anchor_grid 的实际形状计算索引
                auto anchor_grid_shape = anchor_grids_[stage].shape();
                size_t anchor_H        = (anchor_grid_shape.size() >= 3) ? anchor_grid_shape[2] : H;
                size_t anchor_W        = (anchor_grid_shape.size() >= 4) ? anchor_grid_shape[3] : W;

                // 计算 anchor_grid 中的索引（需要缩放）
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
                    (anchor_grid_base < anchor_grid_data.size()) ? anchor_grid_data[anchor_grid_base] : 1.0f;
                float anchor_h =
                    (anchor_grid_base + 1 < anchor_grid_data.size()) ? anchor_grid_data[anchor_grid_base + 1] : 1.0f;

                // if (b == 0 && i == 0 && stage == 0) {
                //     logd("YoloDetect Coordinate Transform (stage {}, box 0): H={}, W={}, num_anchors={}",
                //          stage, H, W, num_anchors_);
                //     logd("grid_shape: {}, anchor_grid_shape: {}", grid_shape.to_string(),
                //     anchor_grid_shape.to_string()); logd("anchor_w={}, anchor_h={}, stride={}", anchor_w, anchor_h,
                //     stride); logd("Before transform: sigmoid_x={}, sigmoid_y={}, sigmoid_w={}, sigmoid_h={}",
                //          sigmoid_data[base_idx + 0], sigmoid_data[base_idx + 1],
                //          sigmoid_data[base_idx + 2], sigmoid_data[base_idx + 3]);
                // }

                // 变换坐标
                // x = (sigmoid_x * 2 + grid_x) * stride
                // y = (sigmoid_y * 2 + grid_y) * stride
                sigmoid_data[base_idx + 0] = (sigmoid_data[base_idx + 0] * 2.0f + grid_x) * stride;
                sigmoid_data[base_idx + 1] = (sigmoid_data[base_idx + 1] * 2.0f + grid_y) * stride;

                // w = (sigmoid_w * 2)^2 * anchor_w
                // h = (sigmoid_h * 2)^2 * anchor_h
                float sigmoid_w            = sigmoid_data[base_idx + 2];
                float sigmoid_h            = sigmoid_data[base_idx + 3];
                sigmoid_data[base_idx + 2] = std::pow(sigmoid_w * 2.0f, 2.0f) * anchor_w;
                sigmoid_data[base_idx + 3] = std::pow(sigmoid_h * 2.0f, 2.0f) * anchor_h;

                // if (b == 0 && i == 0 && stage == 0) {
                //     logd("After transform: x={}, y={}, w={}, h={}",
                //          sigmoid_data[base_idx + 0], sigmoid_data[base_idx + 1],
                //          sigmoid_data[base_idx + 2], sigmoid_data[base_idx + 3]);
                // }
            }
        }

        Tensor transformed_tensor(sigmoid_data, target_shape, dtype(DataType::kFloat32).device(input.device()));
        stage_outputs.push_back(transformed_tensor);
    }

    // 步骤5：拼接所有阶段的输出
    // 使用 cat 操作在维度1上拼接
    if (stage_outputs.size() == 1)
    {
        return std::vector<Tensor>{std::move(stage_outputs[0])};
    }

    auto cat_op        = std::make_shared<functional::Cat>(1);  // 在维度1上拼接
    auto final_outputs = cat_op->forward(stage_outputs);

    return final_outputs;
}

std::vector<Tensor> YoloDetect::backward(const std::vector<Tensor> &gys)
{
    // YOLO Detect 的 backward 实现（暂时不实现，因为主要用于推理）
    THROW_RUNTIME_ERROR("YoloDetect backward is not implemented");
    return std::vector<Tensor>{};
}

Tensor custom_yolo_detect(const std::vector<Tensor> &xs,
                          int32_t stages,
                          int32_t num_classes,
                          int32_t num_anchors,
                          std::vector<float> strides,
                          std::vector<Tensor> anchor_grids,
                          std::vector<Tensor> grids,
                          std::vector<Tensor> conv_weights,
                          std::vector<Tensor> conv_biases)
{
    auto op =
        std::make_shared<YoloDetect>(stages, num_classes, num_anchors, std::move(strides), std::move(anchor_grids),
                                     std::move(grids), std::move(conv_weights), std::move(conv_biases));
    return (*op)(xs)[0];
}

}  // namespace functional
}  // namespace origin
