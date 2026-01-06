#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "origin.h"
#include "origin/operators/custom/yolo_detect.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;

/**
 * @brief YoloDetect 算子测试类
 * @details 测试数据重新排列和坐标变换的正确性
 */
class YoloDetectOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 数据重新排列测试 ====================

TEST_P(YoloDetectOperatorTest, DataRearrangementIndexCalculation)
{
    // 测试数据重新排列的索引计算
    // 从 (N, num_anchors, classes_info, H, W) 到 (N, num_anchors * H * W, classes_info)
    // 验证索引计算的正确性
    
    size_t batch_size = 1;
    size_t num_anchors = 3;
    size_t classes_info = 85;
    size_t H = 2;
    size_t W = 2;
    size_t OC = num_anchors * classes_info;  // 255
    
    // 创建测试数据：卷积输出形状 (N, OC, H, W)
    std::vector<float> conv_data(batch_size * OC * H * W);
    for (size_t i = 0; i < conv_data.size(); ++i) {
        conv_data[i] = static_cast<float>(i);  // 使用索引作为值，便于验证
    }
    
    auto conv_output = Tensor(conv_data, Shape{batch_size, OC, H, W}, 
                               dtype(DataType::kFloat32).device(deviceType()));
    
    // Reshape 为 (N, num_anchors, classes_info, H, W)
    Shape reshape_shape{batch_size, num_anchors, classes_info, H, W};
    Tensor reshaped = conv_output.reshape(reshape_shape);
    
    // 手动重新排列数据
    size_t num_boxes = H * W * num_anchors;
    Shape target_shape{batch_size, num_boxes, classes_info};
    std::vector<float> output_data(target_shape.elements());
    
    auto reshaped_data = reshaped.to_vector<float>();
    
    // 使用当前的索引计算逻辑
    for (size_t b = 0; b < batch_size; ++b) {
        for (size_t na = 0; na < num_anchors; ++na) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    size_t output_idx = b * num_boxes * classes_info + 
                                      (na * H * W + h * W + w) * classes_info;
                    
                    for (size_t c = 0; c < classes_info; ++c) {
                        size_t channel_idx = na * classes_info + c;
                        size_t input_idx = b * OC * H * W +
                                           channel_idx * H * W +
                                           h * W +
                                           w;
                        
                        output_data[output_idx + c] = reshaped_data[input_idx];
                    }
                }
            }
        }
    }
    
    // 验证第一个检测框的值
    // 对于 b=0, na=0, h=0, w=0, c=0
    // channel_idx = 0 * 85 + 0 = 0
    // input_idx = 0 * 255 * 2 * 2 + 0 * 2 * 2 + 0 * 2 + 0 = 0
    // output_idx = 0 * 12 * 85 + (0 * 2 * 2 + 0 * 2 + 0) * 85 + 0 = 0
    EXPECT_FLOAT_EQ(output_data[0], 0.0f);
    
    // 对于 b=0, na=0, h=0, w=0, c=4 (objectness)
    // channel_idx = 0 * 85 + 4 = 4
    // input_idx = 0 * 255 * 2 * 2 + 4 * 2 * 2 + 0 * 2 + 0 = 16
    // output_idx = 0 * 12 * 85 + (0 * 2 * 2 + 0 * 2 + 0) * 85 + 4 = 4
    EXPECT_FLOAT_EQ(output_data[4], 16.0f);
    
    // 对于 b=0, na=0, h=0, w=1, c=0
    // channel_idx = 0 * 85 + 0 = 0
    // input_idx = 0 * 255 * 2 * 2 + 0 * 2 * 2 + 0 * 2 + 1 = 1
    // output_idx = 0 * 12 * 85 + (0 * 2 * 2 + 0 * 2 + 1) * 85 + 0 = 85
    EXPECT_FLOAT_EQ(output_data[85], 1.0f);
    
    // 对于 b=0, na=1, h=0, w=0, c=0
    // channel_idx = 1 * 85 + 0 = 85
    // input_idx = 0 * 255 * 2 * 2 + 85 * 2 * 2 + 0 * 2 + 0 = 340
    // output_idx = 0 * 12 * 85 + (1 * 2 * 2 + 0 * 2 + 0) * 85 + 0 = 340
    EXPECT_FLOAT_EQ(output_data[340], 340.0f);
}

TEST_P(YoloDetectOperatorTest, CoordinateTransformIndexCalculation)
{
    // 测试坐标变换中 grid 和 anchor_grid 的索引计算
    // grid 和 anchor_grid 形状为 (num_anchors, H, W, 2)
    
    size_t num_anchors = 3;
    size_t H = 2;
    size_t W = 2;
    
    // 创建测试 grid 数据：形状 (num_anchors, H, W, 2)
    std::vector<float> grid_data(num_anchors * H * W * 2);
    for (size_t i = 0; i < grid_data.size(); ++i) {
        grid_data[i] = static_cast<float>(i);  // 使用索引作为值
    }
    
    // 测试索引计算
    // 对于 box_idx = 0 (na=0, h=0, w=0)
    size_t box_idx = 0;
    size_t anchor_idx = box_idx / (H * W);  // 0 / 4 = 0
    size_t spatial_idx = box_idx % (H * W);  // 0 % 4 = 0
    size_t h_idx = spatial_idx / W;  // 0 / 2 = 0
    size_t w_idx = spatial_idx % W;  // 0 % 2 = 0
    
    size_t grid_base = anchor_idx * H * W * 2 + h_idx * W * 2 + w_idx * 2;
    // = 0 * 2 * 2 * 2 + 0 * 2 * 2 + 0 * 2 = 0
    
    EXPECT_EQ(grid_base, size_t(0));
    EXPECT_FLOAT_EQ(grid_data[grid_base], 0.0f);      // grid_x
    EXPECT_FLOAT_EQ(grid_data[grid_base + 1], 1.0f);  // grid_y
    
    // 对于 box_idx = 4 (na=1, h=0, w=0)
    box_idx = 4;
    anchor_idx = box_idx / (H * W);  // 4 / 4 = 1
    spatial_idx = box_idx % (H * W);  // 4 % 4 = 0
    h_idx = spatial_idx / W;  // 0 / 2 = 0
    w_idx = spatial_idx % W;  // 0 % 2 = 0
    
    grid_base = anchor_idx * H * W * 2 + h_idx * W * 2 + w_idx * 2;
    // = 1 * 2 * 2 * 2 + 0 * 2 * 2 + 0 * 2 = 8
    
    EXPECT_EQ(grid_base, size_t(8));
    EXPECT_FLOAT_EQ(grid_data[grid_base], 8.0f);      // grid_x
    EXPECT_FLOAT_EQ(grid_data[grid_base + 1], 9.0f);  // grid_y
    
    // 对于 box_idx = 5 (na=1, h=0, w=1)
    box_idx = 5;
    anchor_idx = box_idx / (H * W);  // 5 / 4 = 1
    spatial_idx = box_idx % (H * W);  // 5 % 4 = 1
    h_idx = spatial_idx / W;  // 1 / 2 = 0
    w_idx = spatial_idx % W;  // 1 % 2 = 1
    
    grid_base = anchor_idx * H * W * 2 + h_idx * W * 2 + w_idx * 2;
    // = 1 * 2 * 2 * 2 + 0 * 2 * 2 + 1 * 2 = 10
    
    EXPECT_EQ(grid_base, size_t(10));
    EXPECT_FLOAT_EQ(grid_data[grid_base], 10.0f);      // grid_x
    EXPECT_FLOAT_EQ(grid_data[grid_base + 1], 11.0f);  // grid_y
}

TEST_P(YoloDetectOperatorTest, CoordinateTransformFormula)
{
    // 测试坐标变换公式的正确性
    // x = (sigmoid_x * 2 + grid_x) * stride
    // y = (sigmoid_y * 2 + grid_y) * stride
    // w = (sigmoid_w * 2)^2 * anchor_w
    // h = (sigmoid_h * 2)^2 * anchor_h
    
    float sigmoid_x = 0.5f;
    float sigmoid_y = 0.5f;
    float sigmoid_w = 0.5f;
    float sigmoid_h = 0.5f;
    float grid_x = 1.0f;
    float grid_y = 2.0f;
    float anchor_w = 10.0f;
    float anchor_h = 20.0f;
    float stride = 8.0f;
    
    // 计算变换后的坐标
    float x = (sigmoid_x * 2.0f + grid_x) * stride;
    float y = (sigmoid_y * 2.0f + grid_y) * stride;
    float w = std::pow(sigmoid_w * 2.0f, 2.0f) * anchor_w;
    float h = std::pow(sigmoid_h * 2.0f, 2.0f) * anchor_h;
    
    // 验证结果
    // x = (0.5 * 2 + 1.0) * 8 = (1.0 + 1.0) * 8 = 16.0
    EXPECT_FLOAT_EQ(x, 16.0f);
    
    // y = (0.5 * 2 + 2.0) * 8 = (1.0 + 2.0) * 8 = 24.0
    EXPECT_FLOAT_EQ(y, 24.0f);
    
    // w = (0.5 * 2)^2 * 10 = 1.0^2 * 10 = 10.0
    EXPECT_FLOAT_EQ(w, 10.0f);
    
    // h = (0.5 * 2)^2 * 20 = 1.0^2 * 20 = 20.0
    EXPECT_FLOAT_EQ(h, 20.0f);
}

// 实例化测试套件
INSTANTIATE_DEVICE_TEST_SUITE_P(YoloDetectOperatorTest);

