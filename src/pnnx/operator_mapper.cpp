// 简化的算子映射器实现
#include "origin/pnnx/operator_mapper.h"
#include "origin/pnnx/pnnx_node.h"
#include "origin/operators/conv/conv2d.h"
#include "origin/operators/conv/max_pool2d.h"
#include "origin/operators/activation/silu.h"
#include "origin/operators/nn/upsample.h"
#include "origin/operators/nn/cat.h"
#include "origin/operators/nn/identity.h"
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
    else if (type == "models.yolo.Detect")
    {
        // YOLO 检测层：暂时使用 Identity 算子，直接传递输入
        // TODO: 实现完整的 YOLO 后处理（NMS 等）
        return std::make_shared<Identity>();
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
    // TODO: 实现权重加载
    THROW_RUNTIME_ERROR("Weight loading not implemented yet");
    return Tensor();
}

}  // namespace pnnx
}  // namespace origin

