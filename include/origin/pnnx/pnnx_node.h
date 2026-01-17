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
 */
struct Parameter
{
    int type = 0;  // 0=null, 1=bool, 2=int, 3=float, 4=string, 5=int_array, 6=float_array
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
 * @brief PNNX 属性（权重数据，简化版本）
 */
struct Attribute
{
    int type = 1;  // 1=f32
    std::vector<int> shape;
    std::vector<float> data;  // 权重数据（float 格式）

    Attribute() = default;
};

/**
 * @brief PNNX 运行时节点
 * @details 表示计算图中的一个节点，包含算子、输入输出连接等信息
 */
class PNNXNode
{
public:
    std::string name;              // 节点名称
    std::string type;              // 算子类型（如 nn.Conv2d, nn.SiLU）
    std::shared_ptr<Operator> op;  // 对应的 origindl Operator

    // 输入输出连接
    std::vector<std::string> input_names;   // 输入节点名称列表
    std::vector<std::string> output_names;  // 输出节点名称列表

    // 参数和属性（从 pnnx::Operator 转换而来）
    std::map<std::string, Parameter> params;      // 算子参数
    std::map<std::string, Attribute> attributes;  // 权重数据

    // 形状信息（用于验证）
    std::map<int, std::vector<int>> shapes;  // #0=shape, #1=shape

    // 执行顺序
    int execution_order = -1;

    // 输入输出 Tensor（运行时填充）
    std::map<std::string, Tensor> input_tensors;  // 输入 Tensor，key 为提供者节点名
    std::vector<Tensor> output_tensors;           // 输出 Tensor

    PNNXNode() = default;
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_NODE_H__
