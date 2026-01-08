#ifndef __ORIGIN_DL_PNNX_PARSER_H__
#define __ORIGIN_DL_PNNX_PARSER_H__

#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include "pnnx_node.h"

namespace origin
{
namespace pnnx
{

/**
 * @brief PNNX 文件解析器
 * @details 解析 .param 和 .bin 文件，构建节点列表
 */
class PNNXParser
{
public:
    /**
     * @brief 解析 PNNX 模型文件
     * @param param_path .param 文件路径
     * @param bin_path .bin 文件路径
     * @return 解析后的节点列表
     */
    static std::vector<std::shared_ptr<PNNXNode>> parse(const std::string &param_path,
                                                        const std::string &bin_path);

private:
    /**
     * @brief 解析 .param 文件
     */
    static void parse_param_file(const std::string &param_path,
                                  std::vector<std::shared_ptr<PNNXNode>> &nodes);

    /**
     * @brief 解析一行算子定义
     */
    static void parse_operator_line(const std::string &line,
                                     std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析参数（key=value 格式）
     */
    static void parse_parameter(const std::string &key, const std::string &value,
                                 std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析属性（@key=value 格式，权重数据）
     */
    static void parse_attribute(const std::string &key, const std::string &value,
                                 std::shared_ptr<PNNXNode> &node);

    /**
     * @brief 解析形状信息（#key=value 格式）
     */
    static void parse_shape(const std::string &key, const std::string &value,
                            std::shared_ptr<PNNXNode> &node);


    /**
     * @brief 解析形状字符串，如 "(32,3,6,6)" -> [32,3,6,6]
     */
    static std::vector<int> parse_shape_string(const std::string &shape_str);

    /**
     * @brief 从 .bin 文件加载权重数据
     */
    static void load_weights(const std::string &bin_path,
                             std::vector<std::shared_ptr<PNNXNode>> &nodes);

};

}  // namespace pnnx
}  // namespace origin

#endif  // __ORIGIN_DL_PNNX_PARSER_H__

