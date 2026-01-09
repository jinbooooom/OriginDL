#ifndef __ORIGIN_DL_YOLO_DETECT_H__
#define __ORIGIN_DL_YOLO_DETECT_H__

#include "../../core/operator.h"
#include "../../core/tensor.h"
#include <vector>
#include <memory>
#include <cstdint>

namespace origin
{
namespace functional
{

/**
 * @brief YOLO Detect 算子：YOLOv5 检测层
 * 
 * 输入：
 * - 3个特征图，形状分别为 (N, C1, H1, W1), (N, C2, H2, W2), (N, C3, H3, W3)
 *   通常为 (N, 128, 80, 80), (N, 256, 40, 40), (N, 512, 20, 20)
 * 
 * 输出：
 * - 检测结果，形状为 (N, num_boxes, 85)
 *   其中 85 = 4(bbox坐标) + 1(objectness) + 80(类别分数)
 *   num_boxes = 3 * (80*80 + 40*40 + 20*20) = 25200
 */
class YoloDetect : public Operator
{
public:
    /**
     * @brief 构造函数
     * @param stages 检测阶段数（通常为3）
     * @param num_classes 类别数（COCO为80）
     * @param num_anchors anchor数量（通常为3）
     * @param strides 每个阶段的stride
     * @param anchor_grids anchor grid数据
     * @param grids grid数据
     * @param conv_weights 卷积权重（3个阶段）
     * @param conv_biases 卷积偏置（3个阶段）
     */
    YoloDetect(int32_t stages, 
               int32_t num_classes, 
               int32_t num_anchors,
               std::vector<float> strides,
               std::vector<Tensor> anchor_grids,
               std::vector<Tensor> grids,
               std::vector<Tensor> conv_weights,
               std::vector<Tensor> conv_biases);

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    int32_t stages_;
    int32_t num_classes_;
    int32_t num_anchors_;
    std::vector<float> strides_;
    std::vector<Tensor> anchor_grids_;
    std::vector<Tensor> grids_;
    std::vector<Tensor> conv_weights_;
    std::vector<Tensor> conv_biases_;
};

/**
 * @brief 函数式接口：YOLO Detect 算子
 * @param xs 输入特征图列表，每个特征图形状为 (N, C, H, W)
 * @param stages 检测阶段数（通常为3）
 * @param num_classes 类别数（COCO为80）
 * @param num_anchors anchor数量（通常为3）
 * @param strides 每个阶段的stride
 * @param anchor_grids anchor grid数据
 * @param grids grid数据
 * @param conv_weights 卷积权重（3个阶段）
 * @param conv_biases 卷积偏置（3个阶段）
 * @return 检测结果张量，形状为 (N, num_boxes, 85)
 */
Tensor custom_yolo_detect(const std::vector<Tensor> &xs,
                          int32_t stages,
                          int32_t num_classes,
                          int32_t num_anchors,
                          std::vector<float> strides,
                          std::vector<Tensor> anchor_grids,
                          std::vector<Tensor> grids,
                          std::vector<Tensor> conv_weights,
                          std::vector<Tensor> conv_biases);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_YOLO_DETECT_H__

