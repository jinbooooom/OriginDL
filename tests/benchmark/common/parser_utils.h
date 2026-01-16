#ifndef __BENCHMARK_COMMON_PARSER_UTILS_H__
#define __BENCHMARK_COMMON_PARSER_UTILS_H__

#include <string>
#include <sstream>
#include <vector>
#include "origin/mat/shape.h"
#include "origin/mat/basic_types.h"

/**
 * @brief 解析shape字符串，例如 "100,100" -> Shape({100, 100})
 * @param shape_str shape字符串，例如 "100,100" 或 "1000,1000"
 * @return 解析后的Shape对象，如果解析失败返回空Shape
 */
inline origin::Shape parse_shape_string(const std::string &shape_str)
{
    std::vector<size_t> dims;
    std::stringstream ss(shape_str);
    std::string item;
    
    while (std::getline(ss, item, ','))
    {
        try
        {
            int dim = std::stoi(item);
            if (dim < 0)
            {
                return origin::Shape({});  // 返回空shape表示解析失败
            }
            dims.push_back(static_cast<size_t>(dim));
        }
        catch (const std::exception &)
        {
            return origin::Shape({});  // 返回空shape表示解析失败
        }
    }
    
    return origin::Shape(dims);
}

/**
 * @brief 解析多个shape字符串，例如 "100,200:200,50" -> [Shape({100, 200}), Shape({200, 50})]
 * @param shapes_str shape字符串，用冒号分隔多个shape，例如 "100,200:200,50"
 * @return 解析后的Shape对象列表，如果解析失败返回空列表
 */
inline std::vector<origin::Shape> parse_multiple_shapes_string(const std::string &shapes_str)
{
    std::vector<origin::Shape> shapes;
    std::stringstream ss(shapes_str);
    std::string shape_item;
    
    while (std::getline(ss, shape_item, ':'))
    {
        origin::Shape shape = parse_shape_string(shape_item);
        if (shape.ndims() == 0)
        {
            return std::vector<origin::Shape>();  // 返回空列表表示解析失败
        }
        shapes.push_back(shape);
    }
    
    return shapes;
}

#endif  // __BENCHMARK_COMMON_PARSER_UTILS_H__
