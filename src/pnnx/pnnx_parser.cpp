// 简化的 PNNX 解析器实现
#include "origin/pnnx/pnnx_parser.h"
#include "origin/pnnx/pnnx_node.h"
#include "origin/pnnx/internal/store_zip.hpp"
#include "origin/utils/exception.h"
#include "origin/core/tensor.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <cstring>

namespace origin
{
namespace pnnx
{

std::vector<std::shared_ptr<PNNXNode>> PNNXParser::parse(const std::string &param_path,
                                                          const std::string &bin_path)
{
    std::vector<std::shared_ptr<PNNXNode>> nodes;
    
    // 解析 .param 文件
    parse_param_file(param_path, nodes);
    
    // 解析 .bin 文件加载权重
    if (!bin_path.empty())
    {
        load_weights(bin_path, nodes);
    }
    
    return nodes;
}

void PNNXParser::parse_param_file(const std::string &param_path,
                                    std::vector<std::shared_ptr<PNNXNode>> &nodes)
{
    std::ifstream file(param_path);
    if (!file.is_open())
    {
        THROW_RUNTIME_ERROR("Failed to open param file: {}", param_path);
    }
    
    // 读取 magic number
    std::string line;
    std::getline(file, line);
    // 去除前后空白字符
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    
    if (line.empty())
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
    
    if (magic != 7767517)
    {
        THROW_RUNTIME_ERROR("Invalid PNNX param file, magic number: {}", magic);
    }
    
    // 读取算子数量和操作数数量
    std::getline(file, line);
    std::istringstream iss(line);
    int operator_count, operand_count;
    iss >> operator_count >> operand_count;
    
    // 解析每个算子
    for (int i = 0; i < operator_count; ++i)
    {
        std::getline(file, line);
        if (line.empty()) continue;
        
        auto node = std::make_shared<PNNXNode>();
        parse_operator_line(line, node);
        nodes.push_back(node);
    }
}

void PNNXParser::parse_operator_line(const std::string &line, std::shared_ptr<PNNXNode> &node)
{
    std::istringstream iss(line);
    
    // 读取基本字段：type name input_count output_count
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
    
    // 解析参数和属性
    std::string token;
    while (iss >> token)
    {
        if (token.empty()) continue;
        
        // 解析 key=value 格式
        size_t eq_pos = token.find('=');
        if (eq_pos == std::string::npos) continue;
        
        std::string key = token.substr(0, eq_pos);
        std::string value = token.substr(eq_pos + 1);
        
        if (key[0] == '@')
        {
            // 属性（权重）：@weight=(32,3,6,6)f32
            parse_attribute(key.substr(1), value, node);
        }
        else if (key[0] == '#')
        {
            // 形状信息：#0=(8,3,640,640)f32
            parse_shape(key, value, node);
        }
        else
        {
            // 参数：bias=True, stride=(2,2) 等
            parse_parameter(key, value, node);
        }
    }
}

void PNNXParser::parse_parameter(const std::string &key, const std::string &value,
                                  std::shared_ptr<PNNXNode> &node)
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
        size_t paren_start = value.find('(');
        size_t paren_end = value.find(')');
        
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
                param = Parameter(arr);
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

void PNNXParser::parse_attribute(const std::string &key, const std::string &value,
                                  std::shared_ptr<PNNXNode> &node)
{
    Attribute attr;
    
    // 解析形状：@weight=(32,3,6,6)f32
    size_t shape_end = value.find(')');
    if (shape_end != std::string::npos)
    {
        std::string shape_str = value.substr(1, shape_end - 1);
        attr.shape = parse_shape_string(shape_str);
    }
    
    // 类型：f32
    if (value.find("f32") != std::string::npos)
    {
        attr.type = 1;
    }
    
    // 权重数据将在 load_weights 中加载
    
    node->attributes[key] = attr;
}

void PNNXParser::parse_shape(const std::string &key, const std::string &value,
                              std::shared_ptr<PNNXNode> &node)
{
    // #0=(8,3,640,640)f32
    size_t shape_start = value.find('(');
    size_t shape_end = value.find(')');
    
    if (shape_start != std::string::npos && shape_end != std::string::npos)
    {
        std::string shape_str = value.substr(shape_start + 1, shape_end - shape_start - 1);
        std::vector<int> shape = parse_shape_string(shape_str);
        
        // 提取索引：#0 -> 0
        try
        {
            int index = std::stoi(key.substr(1));
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

void PNNXParser::load_weights(const std::string &bin_path,
                               std::vector<std::shared_ptr<PNNXNode>> &nodes)
{
    using namespace origin::pnnx::internal;
    
    StoreZipReader szr;
    if (szr.open(bin_path) != 0)
    {
        THROW_RUNTIME_ERROR("Failed to open bin file: {}", bin_path);
        return;
    }
    
    // 遍历所有节点，加载权重数据
    for (auto &node : nodes)
    {
        // 遍历节点的所有属性（权重）
        for (auto &attr_pair : node->attributes)
        {
            const std::string &key = attr_pair.first;
            Attribute &attr = attr_pair.second;
            
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
                THROW_RUNTIME_ERROR("Weight file size mismatch for {}: expected {} bytes, got {} bytes",
                                    filename, bytesize, filesize);
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

