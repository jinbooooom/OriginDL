/**
 * @file pnnx_node.h
 * @brief PNNX 节点与参数/属性数据结构
 * @details 定义 Parameter（标量/数组等算子参数）、Attribute（权重
 * shape+data）、PNNXNode（图中单节点：类型、输入输出名、params、attributes、运行时 input/output_tensors 等）。 由
 * PNNXParser 填充构建时数据，由 PNNXGraph 在 set_inputs/forward 中填充运行时 tensor。被
 * pnnx_parser.h、pnnx_graph.h、operator_mapper.h 引用。
 */
#ifndef __ORIGIN_DL_PNNX_NODE_H__
#define __ORIGIN_DL_PNNX_NODE_H__

#include <map>
#include <memory>
#include <string>
#include <vector>
#include "../core/operator.h"
#include "../core/tensor.h"

namespace origin
{
namespace pnnx
{

/**
 * @brief PNNX 参数类型（简化版本）
 * @details 用于存储 .param 中算子参数（非权重），如 stride、padding、bias 等。
 * 由 PNNXParser::parse_parameter 根据 key=value 解析填充；OperatorMapper 各 create_xxx 按需用 get_int_param 等读取。
 */
struct Parameter
{
    int type = 0;  ///< 0=null, 1=bool, 2=int, 3=float, 4=string, 5=int_array, 6=float_array
    bool b   = false;
    int i    = 0;
    float f  = 0.0f;
    std::string s;
    std::vector<int> ai;
    std::vector<float> af;

    Parameter() = default;
    Parameter(bool _b) : type(1), b(_b) {}
    Parameter(int _i) : type(2), i(_i) {}
    Parameter(float _f) : type(3), f(_f) {}
    Parameter(const std::string &_s) : type(4), s(_s) {}
    Parameter(const std::vector<int> &_ai) : type(5), ai(_ai) {}
    Parameter(const std::vector<float> &_af) : type(6), af(_af) {}
};

/**
 * @brief PNNX 算子属性（由 .bin 存储的数据，简化版本）
 * @details 对应 .param 中 @key=value 形式（如 @weight=(32,3,6,6)f32）。常见的是权重（weight、bias），
 * 也有其它由 .bin 存储的常量（如 YOLO 的 pnnx_5/strides）。shape 由 parse_attribute 解析，data 由 load_weights 从 .bin
 * 读入。
 */
struct Attribute
{
    int type = 1;  ///< 张量数据类型，与 Parameter::type 不同套枚举。PNNX：0=未设置 1=f32 2=f16(可扩展)
    std::vector<int> shape;   ///< 形状，如 weight (out_c,in_c,kH,kW) 或 strides 等一维数组
    std::vector<float> data;  ///< 实际数据（float），由 load_weights 从 .bin 填充

    Attribute() = default;
};

/**
 * @brief PNNX 运行时节点
 * @details 表示计算图中的单个算子节点。构建时：name/type/input_names/output_names、params、attributes、shapes 由
 * PNNXParser 填充； build() 中通过 OperatorMapper::create_operator 创建 op；topological_sort 设置 execution_order。
 * 运行时：set_inputs/forward 填充 input_tensors、output_tensors，边通过“输出名即下游输入名”关联。
 * 特殊类型：pnnx.Input（图入口）、pnnx.Output（图出口，其 input_names 即模型最终输出）。
 */
class PNNXNode
{
public:
    std::string name;  ///< 节点名称，如 "pnnx_input_0"、"model.0.conv"、"model.1.conv"、"pnnx_output_0"
    std::string type;  ///< 算子类型，如 nn.Conv2d, nn.SiLU, pnnx.Input, pnnx.Output, models.yolo.Detect
    std::shared_ptr<Operator>
        op;  ///< 对应的 origin Operator，build() 时由 OperatorMapper 创建；pnnx.Input/pnnx.Output 无 op（特殊节点）

    std::vector<std::string> input_names;  ///< 依赖的上游张量名。对于 pnnx.Input
                                           ///< 节点，input_count=0，此列表为空；其它节点，input_count>=1，指向上游输出
    std::vector<std::string>
        output_names;  ///< 本节点产生的张量名，供下游 input_names 引用。对于 pnnx.Output，output_count=0，此列表为空

    std::map<std::string, Parameter> params;  ///< 算子参数（stride、padding 等），由 parser 从 .param 解析
    std::map<std::string, Attribute> attributes;  ///< 权重（weight、bias 等），shape 在 param 中，数据在 .bin 中加载

    std::map<int, std::vector<int>> shapes;  ///< 形状信息，#0=..., #1=... 等，用于验证或解析输入尺寸

    int execution_order = -1;  ///< 拓扑排序后的执行顺序，forward 时按此顺序执行

    std::map<std::string, Tensor>
        input_tensors;  ///< 运行时：上游传入的 Tensor，key 为 input_names 中的名字。对于 pnnx.Input，此映射为空；对于
                        ///< pnnx.Output，由 propagate_outputs 填充
    std::vector<Tensor>
        output_tensors;  ///< 运行时：本节点 op 的输出，由 propagate_outputs 传给下游。对于 pnnx.Input，由 set_inputs()
                         ///< 直接设置；对于 pnnx.Output，此列表为空（不计算，只收集）

    PNNXNode() = default;
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_NODE_H__
