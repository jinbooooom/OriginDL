#include "origin/operators/custom/yolo_detect.h"
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/operators/shape/cat.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

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
    if (unlikely(stages_ != 3))
    {
        THROW_RUNTIME_ERROR("YoloDetect only supports 3 stages, but got {}", stages_);
    }
    if (unlikely(strides_.size() != static_cast<size_t>(stages_)))
    {
        THROW_RUNTIME_ERROR("YoloDetect: strides size ({}) does not match stages ({})", strides_.size(), stages_);
    }
    if (unlikely(anchor_grids_.size() != static_cast<size_t>(stages_)))
    {
        THROW_RUNTIME_ERROR("YoloDetect: anchor_grids size ({}) does not match stages ({})", anchor_grids_.size(),
                            stages_);
    }
    if (unlikely(grids_.size() != static_cast<size_t>(stages_)))
    {
        THROW_RUNTIME_ERROR("YoloDetect: grids size ({}) does not match stages ({})", grids_.size(), stages_);
    }
    if (unlikely(conv_weights_.size() != static_cast<size_t>(stages_)))
    {
        THROW_RUNTIME_ERROR("YoloDetect: conv_weights size ({}) does not match stages ({})", conv_weights_.size(),
                            stages_);
    }
    if (unlikely(conv_biases_.size() != static_cast<size_t>(stages_)))
    {
        THROW_RUNTIME_ERROR("YoloDetect: conv_biases size ({}) does not match stages ({})", conv_biases_.size(),
                            stages_);
    }
}

std::vector<Tensor> YoloDetect::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != static_cast<size_t>(stages_)))
    {
        THROW_RUNTIME_ERROR("YoloDetect forward: expected {} inputs (stages), but got {}", stages_, xs.size());
    }

    // 存储每个阶段的输出
    std::vector<Tensor> stage_outputs;

    // 对每个阶段进行处理
    for (int32_t stage = 0; stage < stages_; ++stage)
    {
        const Tensor &input = xs[stage];
        auto input_shape    = input.shape();

        // 检查输入形状：应该是 (N, C, H, W)
        if (unlikely(input_shape.size() != 4))
        {
            THROW_RUNTIME_ERROR("YoloDetect forward: input {} must be 4D (N, C, H, W), but got shape {}", stage,
                                input_shape.to_string());
        }

        // 确保所有张量都在同一设备上（与输入张量相同）
        Device input_device = input.device();
        Tensor conv_weight_on_device =
            conv_weights_[stage].device() == input_device ? conv_weights_[stage] : conv_weights_[stage].to(input_device);
        Tensor conv_bias_on_device =
            conv_biases_[stage].device() == input_device ? conv_biases_[stage] : conv_biases_[stage].to(input_device);
        Tensor grid_on_device =
            grids_[stage].device() == input_device ? grids_[stage] : grids_[stage].to(input_device);
        Tensor anchor_grid_on_device = anchor_grids_[stage].device() == input_device ? anchor_grids_[stage]
                                                                                      : anchor_grids_[stage].to(input_device);

        // 使用 mat 层的 yolo_detect_forward 实现
        const Mat &input_mat         = mat(input);
        const Mat &conv_weight_mat   = mat(conv_weight_on_device);
        const Mat *conv_bias_mat     = conv_bias_on_device.shape().elements() > 0 ? &mat(conv_bias_on_device) : nullptr;
        const Mat &grid_mat         = mat(grid_on_device);
        const Mat &anchor_grid_mat   = mat(anchor_grid_on_device);

        auto stage_result = input_mat.yolo_detect_forward(
            conv_weight_mat,
            conv_bias_mat,
            grid_mat,
            anchor_grid_mat,
            strides_[stage],
            num_anchors_,
            num_classes_
        );

        Tensor stage_tensor = convert_mat_to_tensor(std::move(stage_result));
        stage_outputs.push_back(std::move(stage_tensor));
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
