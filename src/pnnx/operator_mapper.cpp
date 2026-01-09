// 简化的算子映射器实现
#include "origin/pnnx/operator_mapper.h"
#include "origin/pnnx/pnnx_node.h"
#include "origin/operators/conv/conv2d.h"
#include "origin/operators/pooling/max_pool2d.h"
#include "origin/operators/activation/silu.h"
#include "origin/operators/nn/upsample.h"
#include "origin/operators/shape/cat.h"
#include "origin/operators/nn/identity.h"
#include "origin/operators/pooling/adaptive_avg_pool2d.h"
#include "origin/operators/shape/flatten.h"
#include "origin/operators/custom/linear.h"
#include "origin/operators/custom/yolo_detect.h"
#include "origin/core/operator.h"
#include "origin/utils/exception.h"
#include <memory>
#include <string>

namespace origin
{
namespace pnnx
{

std::shared_ptr<Operator> OperatorMapper::create_operator(std::shared_ptr<PNNXNode> node)
{
    const std::string &type = node->type;
    
    if (type == "nn.Conv2d")
    {
        return create_conv2d(node);
    }
    else if (type == "nn.SiLU")
    {
        return create_silu(node);
    }
    else if (type == "nn.ReLU")
    {
        return create_relu(node);
    }
    else if (type == "nn.Upsample")
    {
        return create_upsample(node);
    }
    else if (type == "nn.MaxPool2d")
    {
        return create_max_pool2d(node);
    }
    else if (type == "pnnx.Expression")
    {
        return create_expression(node);
    }
    else if (type == "torch.cat")
    {
        return create_cat(node);
    }
    else if (type == "nn.AdaptiveAvgPool2d")
    {
        return create_adaptive_avg_pool2d(node);
    }
    else if (type == "torch.flatten")
    {
        return create_flatten(node);
    }
    else if (type == "nn.Linear")
    {
        return create_linear(node);
    }
    else if (type == "models.yolo.Detect")
    {
        return create_yolo_detect(node);
    }
    else
    {
        THROW_RUNTIME_ERROR("Unsupported operator type: {}", type);
        return nullptr;
    }
}

std::shared_ptr<Operator> OperatorMapper::create_conv2d(std::shared_ptr<PNNXNode> node)
{
    // 从参数中提取 Conv2d 的参数
    auto &params = node->params;
    
    // 获取参数值
    std::pair<int, int> stride = get_int_pair_param(params, "stride", {1, 1});
    std::pair<int, int> pad = get_int_pair_param(params, "padding", {0, 0});
    
    // 创建 Conv2dOp
    return std::make_shared<Conv2dOp>(stride, pad);
}

std::shared_ptr<Operator> OperatorMapper::create_silu(std::shared_ptr<PNNXNode> node)
{
    // SiLU 激活函数：x * sigmoid(x)
    return std::make_shared<SiLU>();
}

std::shared_ptr<Operator> OperatorMapper::create_relu(std::shared_ptr<PNNXNode> node)
{
    // ReLU 激活函数：max(0, x)
    return std::make_shared<ReLU>();
}

std::shared_ptr<Operator> OperatorMapper::create_adaptive_avg_pool2d(std::shared_ptr<PNNXNode> node)
{
    auto &params = node->params;
    
    // 获取 output_size 参数，例如 output_size=(1,1)
    std::pair<int, int> output_size = get_int_pair_param(params, "output_size", {1, 1});
    
    return std::make_shared<AdaptiveAvgPool2d>(output_size);
}

std::shared_ptr<Operator> OperatorMapper::create_flatten(std::shared_ptr<PNNXNode> node)
{
    auto &params = node->params;
    
    // 获取 start_dim 和 end_dim 参数
    int start_dim = get_int_param(params, "start_dim", 1);
    int end_dim = get_int_param(params, "end_dim", -1);
    
    return std::make_shared<FlattenOp>(start_dim, end_dim);
}

std::shared_ptr<Operator> OperatorMapper::create_linear(std::shared_ptr<PNNXNode> node)
{
    auto &params = node->params;
    
    // 获取 in_features 和 out_features 参数
    int in_features = get_int_param(params, "in_features", 0);
    int out_features = get_int_param(params, "out_features", 0);
    
    // 检查是否有 bias 参数
    bool use_bias = true;
    if (params.find("bias") != params.end())
    {
        // bias 可能是 bool 类型
        if (params.at("bias").type == 0)  // bool type
        {
            use_bias = params.at("bias").b;
        }
    }
    
    if (in_features == 0 || out_features == 0)
    {
        THROW_RUNTIME_ERROR("Linear operator: in_features and out_features must be specified");
    }
    
    return std::make_shared<LinearOp>(in_features, out_features, use_bias);
}

std::shared_ptr<Operator> OperatorMapper::create_upsample(std::shared_ptr<PNNXNode> node)
{
    // Upsample 上采样
    auto &params = node->params;
    
    // 获取 scale_factor 或 size
    std::string mode = "nearest";  // 默认最近邻
    if (params.find("mode") != params.end())
    {
        mode = params.at("mode").s;
    }
    
    // 检查是否有 scale_factor
    if (params.find("scale_factor") != params.end())
    {
        auto scale = get_int_pair_param(params, "scale_factor", {2, 2});
        float scale_h = static_cast<float>(scale.first);
        float scale_w = static_cast<float>(scale.second);
        std::pair<float, float> scale_pair(scale_h, scale_w);
        return std::make_shared<Upsample>(mode, scale_pair);
    }
    else if (params.find("size") != params.end())
    {
        auto size = get_int_pair_param(params, "size", {0, 0});
        return std::make_shared<Upsample>(mode, size);
    }
    else
    {
        // 默认 2x 上采样
        std::pair<float, float> default_scale(2.0f, 2.0f);
        return std::make_shared<Upsample>(mode, default_scale);
    }
}

std::shared_ptr<Operator> OperatorMapper::create_max_pool2d(std::shared_ptr<PNNXNode> node)
{
    auto &params = node->params;
    
    // 获取 kernel_size, stride, padding
    std::pair<int, int> kernel_size = get_int_pair_param(params, "kernel_size", {2, 2});
    std::pair<int, int> stride = get_int_pair_param(params, "stride", {0, 0});  // 0 表示使用 kernel_size
    std::pair<int, int> pad = get_int_pair_param(params, "padding", {0, 0});
    
    return std::make_shared<MaxPool2d>(kernel_size, stride, pad);
}

std::shared_ptr<Operator> OperatorMapper::create_expression(std::shared_ptr<PNNXNode> node)
{
    // pnnx.Expression：如 add(@0,@1)
    auto &params = node->params;
    
    if (params.find("expr") != params.end())
    {
        std::string expr = params.at("expr").s;
        
        if (expr.find("add") != std::string::npos)
        {
            // 使用 Add 算子
            return std::make_shared<Add>();
        }
        // TODO: 支持其他表达式（mul, sub 等）
    }
    
    THROW_RUNTIME_ERROR("Unsupported expression: {}", 
                        params.find("expr") != params.end() ? params.at("expr").s : "unknown");
    return nullptr;
}

std::shared_ptr<Operator> OperatorMapper::create_cat(std::shared_ptr<PNNXNode> node)
{
    // torch.cat：拼接操作
    auto &params = node->params;
    
    // 获取 dim 参数
    int dim = get_int_param(params, "dim", 0);
    
    return std::make_shared<Cat>(dim);
}

int OperatorMapper::get_int_param(const std::map<std::string, Parameter> &params,
                                    const std::string &key, int default_value)
{
    auto it = params.find(key);
    if (it != params.end() && it->second.type == 2)  // int type
    {
        return it->second.i;
    }
    return default_value;
}

std::pair<int, int> OperatorMapper::get_int_pair_param(const std::map<std::string, Parameter> &params,
                                                         const std::string &key,
                                                         std::pair<int, int> default_value)
{
    auto it = params.find(key);
    if (it != params.end() && it->second.type == 5)  // int_array type
    {
        const auto &arr = it->second.ai;
        if (arr.size() >= 2)
        {
            return {arr[0], arr[1]};
        }
        else if (arr.size() == 1)
        {
            return {arr[0], arr[0]};
        }
    }
    return default_value;
}

Tensor OperatorMapper::load_weight_tensor(const Attribute &attr)
{
    // 从 Attribute 加载权重 Tensor
    if (attr.data.empty() || attr.shape.empty())
    {
        THROW_RUNTIME_ERROR("Attribute data or shape is empty");
    }
    
    std::vector<size_t> shape_vec;
    for (int dim : attr.shape)
    {
        shape_vec.push_back(static_cast<size_t>(dim));
    }
    Shape weight_shape(shape_vec);
    
    // 使用 CPU 设备（权重通常在 CPU 上）
    Device device(DeviceType::kCPU);
    
    return Tensor(attr.data, weight_shape, dtype(DataType::kFloat32).device(device));
}

std::shared_ptr<Operator> OperatorMapper::create_yolo_detect(std::shared_ptr<PNNXNode> node)
{
    auto &attrs = node->attributes;
    
    // 读取 strides (pnnx_5)
    if (attrs.find("pnnx_5") == attrs.end())
    {
        THROW_RUNTIME_ERROR("YoloDetect: cannot find pnnx_5 (strides) attribute");
    }
    auto &strides_attr = attrs.at("pnnx_5");
    std::vector<float> strides = strides_attr.data;
    int32_t stages = static_cast<int32_t>(strides.size());
    
    if (stages != 3)
    {
        THROW_RUNTIME_ERROR("YoloDetect: only supports 3 stages, but got {}", stages);
    }
    
    // 读取 anchor_grids (pnnx_4, pnnx_2, pnnx_0) - 倒序，与 KuiperInferGitee 一致
    // 
    // 说明：
    // 1. 通过修正 anchor_grid 的加载顺序，修复了 YOLOv5 检测框绘制不正确的问题
    // 2. 将加载顺序从 [pnnx_0, pnnx_2, pnnx_4] 改为 [pnnx_4, pnnx_2, pnnx_0]，与 KuiperInferGitee 保持一致
    // 
    // 原因：KuiperInferGitee 使用倒序加载 anchor_grid，如果 origindl 使用正序会导致坐标变换时使用错误的 anchor 值，
    // 从而产生不正确的检测框坐标（宽高值错误），最终导致检测框绘制位置不正确。
    std::vector<Tensor> anchor_grids;
    std::vector<int> anchor_indices = {4, 2, 0};
    for (int idx : anchor_indices)
    {
        std::string attr_name = "pnnx_" + std::to_string(idx);
        if (attrs.find(attr_name) == attrs.end())
        {
            THROW_RUNTIME_ERROR("YoloDetect: cannot find {} (anchor_grid) attribute", attr_name);
        }
        anchor_grids.push_back(load_weight_tensor(attrs.at(attr_name)));
    }
    
    // 读取 grids (pnnx_6, pnnx_3, pnnx_1)
    std::vector<Tensor> grids;
    std::vector<int> grid_indices = {6, 3, 1};
    for (int idx : grid_indices)
    {
        std::string attr_name = "pnnx_" + std::to_string(idx);
        if (attrs.find(attr_name) == attrs.end())
        {
            THROW_RUNTIME_ERROR("YoloDetect: cannot find {} (grid) attribute", attr_name);
        }
        grids.push_back(load_weight_tensor(attrs.at(attr_name)));
    }
    
    // 读取卷积权重和偏置 (m.0.weight/bias, m.1.weight/bias, m.2.weight/bias)
    std::vector<Tensor> conv_weights;
    std::vector<Tensor> conv_biases;
    int32_t num_classes = -1;
    int32_t num_anchors = -1;
    
    for (int32_t i = 0; i < stages; ++i)
    {
        std::string weight_name = "m." + std::to_string(i) + ".weight";
        std::string bias_name = "m." + std::to_string(i) + ".bias";
        
        if (attrs.find(weight_name) == attrs.end())
        {
            THROW_RUNTIME_ERROR("YoloDetect: cannot find {} attribute", weight_name);
        }
        if (attrs.find(bias_name) == attrs.end())
        {
            THROW_RUNTIME_ERROR("YoloDetect: cannot find {} attribute", bias_name);
        }
        
        Tensor weight = load_weight_tensor(attrs.at(weight_name));
        Tensor bias = load_weight_tensor(attrs.at(bias_name));
        
        conv_weights.push_back(weight);
        conv_biases.push_back(bias);
        
        // 从第一个权重推断 num_classes 和 num_anchors
        if (i == 0)
        {
            auto weight_shape = weight.shape();
            if (weight_shape.size() != 4)
            {
                THROW_RUNTIME_ERROR("YoloDetect: conv weight must be 4D, but got shape {}", 
                                   weight_shape.to_string());
            }
            int32_t out_channels = static_cast<int32_t>(weight_shape[0]);
            if (num_anchors == -1)
            {
                // 假设 num_anchors = 3（YOLOv5 标准）
                num_anchors = 3;
                if (out_channels % num_anchors != 0)
                {
                    THROW_RUNTIME_ERROR("YoloDetect: out_channels ({}) must be divisible by num_anchors ({})", 
                                       out_channels, num_anchors);
                }
                num_classes = out_channels / num_anchors - 5;  // 5 = 4(bbox) + 1(objectness)
                if (num_classes <= 0)
                {
                    THROW_RUNTIME_ERROR("YoloDetect: invalid num_classes ({})", num_classes);
                }
            }
        }
    }
    
    return std::make_shared<YoloDetect>(stages, num_classes, num_anchors,
                                       std::move(strides),
                                       std::move(anchor_grids),
                                       std::move(grids),
                                       std::move(conv_weights),
                                       std::move(conv_biases));
}

}  // namespace pnnx
}  // namespace origin

