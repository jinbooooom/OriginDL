/**
 * @file pnnx_parser.h
 * @brief PNNX 模型文件解析器声明
 * @details 声明 PNNXParser，负责从 .param（图结构、形状、属性元信息）和 .bin（权重二进制）解析出 PNNXNode 列表。
 * 由 PNNXGraph::init() 调用 parse()；实现见 src/pnnx/pnnx_parser.cpp，权重读取可能依赖 internal/store_zip。
 */
#ifndef __ORIGIN_DL_PNNX_PARSER_H__
#define __ORIGIN_DL_PNNX_PARSER_H__

#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include "pnnx_node.h"

namespace origin
{
namespace pnnx
{

/**
 * @brief PNNX 文件解析器
 * @details
 * 负责从 PNNX 模型文件构建节点列表，供 PNNXGraph::init() 调用。
 * - .param 文件：文本格式。首行 magic 7767517；第二行 operator_count、operand_count；之后每行一个算子。
 *   每行格式：type name input_count output_count [input_names...] [output_names...] [可选 key=value...]。
 *   可选项包括：普通参数（如 stride=(2,2)）、@attr=（如 @weight=(32,3,6,6)f32）、#shape=（如 #0=(8,3,640,640)f32）。
 * - .bin 文件：二进制权重，按 param 中出现的 @ 属性顺序依次读取，填充到各节点的 attributes[].data。
 * 解析后得到 nodes（含 pnnx.Input、pnnx.Output 及所有计算节点），节点间通过 input_names/output_names 在图中连接。
 */
class PNNXParser
{
public:
    /**
     * @brief 解析 PNNX 模型文件
     * @param param_path .param 文件路径
     * @param bin_path .bin 文件路径（可为空则仅解析结构不加载权重）
     * @return 解析后的节点列表，顺序与 param 中算子顺序一致
     */
    static std::vector<std::shared_ptr<PNNXNode>> parse(const std::string &param_path, const std::string &bin_path);

private:
    /**
     * @brief 解析 .param 文件
     * @details 读 magic、算子数量，然后逐行 parse_operator_line 得到 nodes
     */
    static void parse_param_file(const std::string &param_path, std::vector<std::shared_ptr<PNNXNode>> &nodes);

    /**
     * @brief 解析一行算子定义
     * @details 解析 type name input_count output_count、input_names、output_names，以及行内 key=value（交给 parse_parameter/parse_attribute/parse_shape）
     */
    static void parse_operator_line(const std::string &line, std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析普通参数（key=value，非 @ 非 #）
     * @details 支持 bool、int、float、string、(1,2) 等整数数组；结果写入 node->params[key]
     */
    static void parse_parameter(const std::string &key, const std::string &value, std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析属性（@key=value，如 @weight=(32,3,6,6)f32）
     * @details 解析 shape 与 type，写入 node->attributes[key]；实际 data 由 load_weights 从 .bin 填充
     */
    static void parse_attribute(const std::string &key, const std::string &value, std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析形状信息（#key=value，如 #0=(8,3,640,640)f32）
     * @details 将 key 中数字作为索引，value 中括号内形状解析为整数数组，写入 node->shapes[index]
     */
    static void parse_shape(const std::string &key, const std::string &value, std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析形状字符串为整数数组
     * @param shape_str 如 "(32,3,6,6)" 或 "8,3,640,640"
     * @return 如 [32,3,6,6]
     */
    static std::vector<int> parse_shape_string(const std::string &shape_str);

    /**
     * @brief 从 .bin 文件加载权重数据
     * @details 按 nodes 顺序及每个节点 attributes 中需数据的项，从 bin 中依次读取并填充 attributes[].data
     */
    static void load_weights(const std::string &bin_path, std::vector<std::shared_ptr<PNNXNode>> &nodes);
};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_PARSER_H__
