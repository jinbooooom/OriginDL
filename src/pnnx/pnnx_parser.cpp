#include "origin/pnnx/pnnx_parser.h"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "origin/core/tensor.h"
#include "origin/pnnx/internal/store_zip.hpp"
#include "origin/pnnx/pnnx_node.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace pnnx
{

// PNNX 模型文件解析：.param 描述图结构/形状/属性元信息，.bin 存权重。由 PNNXGraph::init() 调用 parse()。
// 入口：先解析 param 得到节点列表（含 pnnx.Input、pnnx.Output 及所有算子），再按节点 attributes 从 bin 加载权重到 data
std::vector<std::shared_ptr<PNNXNode>> PNNXParser::parse(const std::string &param_path, const std::string &bin_path)
{
    std::vector<std::shared_ptr<PNNXNode>> nodes;

    parse_param_file(param_path, nodes);

    if (!bin_path.empty())
    {
        load_weights(bin_path, nodes);
    }

    return nodes;
}

// param 格式：首行 magic 7767517；第二行 operator_count operand_count；之后每行一个算子（见 parse_operator_line）
// 更多格式参考 https://github.com/Tencent/ncnn/wiki/param-and-model-file-structure
// https://github.com/Tencent/ncnn/wiki/operation-param-weight-table
void PNNXParser::parse_param_file(const std::string &param_path, std::vector<std::shared_ptr<PNNXNode>> &nodes)
{
    std::ifstream file(param_path);
    if (unlikely(!file.is_open()))
    {
        THROW_RUNTIME_ERROR("Failed to open param file: {}", param_path);
    }

    std::string line;
    std::getline(file, line);
    // 去除前后空白字符
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);

    if (unlikely(line.empty()))
    {
        THROW_RUNTIME_ERROR("Empty magic number in param file");
    }

    int magic = 0;
    try
    {
        magic = std::stoi(line);
    }
    catch (const std::exception &e)
    {
        THROW_RUNTIME_ERROR("Failed to parse magic number: '{}', error: {}", line, e.what());
    }

    if (magic != 7767517) // PNNX 的 magic number
    {
        THROW_RUNTIME_ERROR("Invalid PNNX param file, magic number: {}", magic);
    }

    std::getline(file, line);
    std::istringstream iss(line);
    int operator_count, operand_count;
    iss >> operator_count >> operand_count;

    for (int i = 0; i < operator_count; ++i)
    {
        std::getline(file, line);
        if (line.empty())
            continue;

        auto node = std::make_shared<PNNXNode>();
        parse_operator_line(line, node);
        nodes.push_back(node);
    }
}

// 每行格式：type name input_count output_count [input_names...] [output_names...] [@attr=...] [#shape=...] [param=...]
// 例如 pnnx.Input 的 #0=(4,3,640,640)f32 即输入形状，yolov5_infer 中解析输入尺寸即用同一格式
void PNNXParser::parse_operator_line(const std::string &line, std::shared_ptr<PNNXNode> &node)
{
    std::istringstream iss(line);

    std::string type, name;
    int input_count, output_count;
    iss >> type >> name >> input_count >> output_count;

    node->type = type;
    node->name = name;

    // 读取输入名称
    for (int i = 0; i < input_count; ++i)
    {
        std::string input_name;
        iss >> input_name;
        node->input_names.push_back(input_name);
    }

    // 读取输出名称
    for (int i = 0; i < output_count; ++i)
    {
        std::string output_name;
        iss >> output_name;
        node->output_names.push_back(output_name);
    }

    // 行内 key=value：@ 为属性（权重，shape 在此解析，data 在 load_weights 从 bin 读）；# 为形状；其余为参数
    std::string token;
    while (iss >> token)
    {
        if (token.empty())
            continue;

        size_t eq_pos = token.find('=');
        if (eq_pos == std::string::npos)
            continue;

        std::string key   = token.substr(0, eq_pos);
        std::string value = token.substr(eq_pos + 1);

        if (key[0] == '@')
        {
            parse_attribute(key.substr(1), value, node);  // @weight=(32,3,6,6)f32 -> attributes["weight"].shape
        }
        else if (key[0] == '#')
        {
            parse_shape(key, value, node);  // #0=(8,3,640,640)f32 -> shapes[0]
        }
        else
        {
            parse_parameter(key, value, node);  // stride=(2,2), bias=True 等 -> params
        }
    }
}

void PNNXParser::parse_parameter(const std::string &key, const std::string &value, std::shared_ptr<PNNXNode> &node)
{
    Parameter param;

    // 解析不同类型的参数值
    if (value == "True" || value == "true")
    {
        param = Parameter(true);
    }
    else if (value == "False" || value == "false")
    {
        param = Parameter(false);
    }
    else if (value.find('(') != std::string::npos && value.find(')') != std::string::npos)
    {
        // 检查是否是纯数字数组格式：(1,1) 或 (2,2)
        // 排除表达式如 add(@0,@1) 或包含非数字字符的情况
        bool is_numeric_array = true;
        size_t paren_start    = value.find('(');
        size_t paren_end      = value.find(')');

        if (paren_start != std::string::npos && paren_end != std::string::npos && paren_end > paren_start)
        {
            std::string content = value.substr(paren_start + 1, paren_end - paren_start - 1);
            // 检查内容是否只包含数字、逗号和空格
            for (char c : content)
            {
                if (c != ',' && c != ' ' && c != '\t' && !std::isdigit(c) && c != '-')
                {
                    is_numeric_array = false;
                    break;
                }
            }

            if (is_numeric_array)
            {
                // 数组格式：(1,1) 或 (2,2)
                std::vector<int> arr = parse_shape_string(value);
                param                = Parameter(arr);
            }
            else
            {
                // 表达式或其他格式，作为字符串处理
                param = Parameter(value);
            }
        }
        else
        {
            // 括号不匹配，作为字符串处理
            param = Parameter(value);
        }
    }
    else
    {
        // 尝试解析为数字
        try
        {
            // 去除前后空白字符
            std::string trimmed_value = value;
            trimmed_value.erase(0, trimmed_value.find_first_not_of(" \t\r\n"));
            trimmed_value.erase(trimmed_value.find_last_not_of(" \t\r\n") + 1);

            if (trimmed_value.empty())
            {
                // 空字符串，跳过
                return;
            }

            if (trimmed_value.find('.') != std::string::npos)
            {
                param = Parameter(std::stof(trimmed_value));
            }
            else
            {
                param = Parameter(std::stoi(trimmed_value));
            }
        }
        catch (const std::exception &e)
        {
            // 解析失败，作为字符串处理
            param = Parameter(value);
        }
    }

    node->params[key] = param;
}

// 解析 @key=value：得到 attr.shape 与 type，data 留空，由 load_weights 从 .bin 按 node+attr 顺序读入
void PNNXParser::parse_attribute(const std::string &key, const std::string &value, std::shared_ptr<PNNXNode> &node)
{
    Attribute attr;

    size_t shape_end = value.find(')');
    if (shape_end != std::string::npos)
    {
        std::string shape_str = value.substr(1, shape_end - 1);
        attr.shape            = parse_shape_string(shape_str);
    }

    if (value.find("f32") != std::string::npos)
    {
        attr.type = 1;
    }

    node->attributes[key] = attr;
}

// #key=value 解析为 node->shapes[索引]，如 #0=(8,3,640,640)f32 -> shapes[0]=[8,3,640,640]
// yolov5_infer 里 get_input_shape_from_param_file 解析的 #0=(...) 格式与本函数一致
void PNNXParser::parse_shape(const std::string &key, const std::string &value, std::shared_ptr<PNNXNode> &node)
{
    size_t shape_start = value.find('(');
    size_t shape_end   = value.find(')');

    if (shape_start != std::string::npos && shape_end != std::string::npos)
    {
        std::string shape_str  = value.substr(shape_start + 1, shape_end - shape_start - 1);
        std::vector<int> shape = parse_shape_string(shape_str);

        // 提取索引：#0 -> 0
        try
        {
            int index           = std::stoi(key.substr(1));
            node->shapes[index] = shape;
        }
        catch (const std::exception &e)
        {
            THROW_RUNTIME_ERROR("Failed to parse shape index from key: '{}', error: {}", key, e.what());
        }
    }
}

std::vector<int> PNNXParser::parse_shape_string(const std::string &shape_str)
{
    std::vector<int> shape;
    std::istringstream iss(shape_str);
    std::string token;

    while (std::getline(iss, token, ','))
    {
        // 去除空格、换行符和括号
        token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\t'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\r'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '\n'), token.end());
        token.erase(std::remove(token.begin(), token.end(), '('), token.end());
        token.erase(std::remove(token.begin(), token.end(), ')'), token.end());

        if (!token.empty())
        {
            try
            {
                shape.push_back(std::stoi(token));
            }
            catch (const std::exception &e)
            {
                THROW_RUNTIME_ERROR("Failed to parse shape dimension: '{}', error: {}", token, e.what());
            }
        }
    }

    return shape;
}

// .bin 为 store-only zip：每个 Attribute 对应 zip 里的一个独立文件，文件名为 "node_name.attr_key"。
// 这里不按 .param 中出现的顺序顺读，而是：
// 1) 先在 open() 时由 StoreZipReader 扫描 zip 目录，记录每个文件的 offset/size；
// 2) load_weights 按节点和 attributes 遍历，按 node->name + "." + key 构造文件名；
// 3) 通过 get_file_size()/read_file() 在内部根据 offset 精确定位并一次性读出该 Attribute 的连续数据到 attr.data。
void PNNXParser::load_weights(const std::string &bin_path, std::vector<std::shared_ptr<PNNXNode>> &nodes)
{
    using namespace origin::pnnx::internal;

    StoreZipReader szr;
    if (szr.open(bin_path) != 0)
    {
        THROW_RUNTIME_ERROR("Failed to open bin file: {}", bin_path);
        return;
    }

    for (auto &node : nodes)
    {
        // 遍历节点的所有属性（权重）
        for (auto &attr_pair : node->attributes)
        {
            const std::string &key = attr_pair.first;
            Attribute &attr        = attr_pair.second;

            // 计算权重数据大小
            if (attr.shape.empty() || attr.type != 1)  // type 1 = f32
            {
                continue;
            }

            size_t size = 1;
            for (int dim : attr.shape)
            {
                size *= dim;
            }

            size_t bytesize = size * sizeof(float);  // f32 = 4 bytes

            // 构建文件名：operator_name.attr_name
            std::string filename = node->name + "." + key;

            // 检查文件是否存在
            size_t filesize = szr.get_file_size(filename);
            if (filesize == 0)
            {
                // 没有这个权重文件，跳过
                continue;
            }

            if (filesize != bytesize)
            {
                THROW_RUNTIME_ERROR("Weight file size mismatch for {}: expected {} bytes, got {} bytes", filename,
                                    bytesize, filesize);
                continue;
            }

            // 读取权重数据
            std::vector<float> weight_data(size);
            if (szr.read_file(filename, (char *)weight_data.data()) != 0)
            {
                THROW_RUNTIME_ERROR("Failed to read weight file: {}", filename);
                continue;
            }

            // 将权重数据存储到 Attribute 中
            attr.data = weight_data;
        }
    }

    szr.close();
}

}  // namespace pnnx
}  // namespace origin
