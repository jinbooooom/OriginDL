#ifndef __BENCHMARK_COMMON_BENCHMARK_FRAMEWORK_H__
#define __BENCHMARK_COMMON_BENCHMARK_FRAMEWORK_H__

#include <string>
#include <vector>
#include "benchmark/common/parser_utils.h"
#include "origin.h"

namespace origin
{
namespace benchmark
{

/**
 * @brief 基准测试配置结构体
 * 包含运行基准测试所需的所有配置信息
 */
struct BenchmarkConfig
{
    std::vector<Shape> shapes;  // 输入张量形状列表（支持多个shape，如matmul需要两个）
    DataType dtype;             // 数据类型
    Device device;              // 设备
    int warmup_cnt;             // 预热次数
    int repeat_cnt;             // 重复测试次数
    bool inplace;               // 是否使用就地操作
};

/**
 * @brief 基准测试框架基类
 * 提供命令行解析、测试循环等通用功能
 * 子类需要实现 run_benchmark() 方法来执行具体的基准测试
 */
class BenchmarkFramework
{
public:
    virtual ~BenchmarkFramework() = default;

    /**
     * @brief 运行基准测试（纯虚函数，由子类实现）
     * @param config 基准测试配置
     * @return 平均执行时间（微秒）
     */
    virtual double run_benchmark(const BenchmarkConfig &config) = 0;

    /**
     * @brief 获取默认测试形状
     * @return 默认形状列表，每个元素是一个shape组合（对于单shape算子，每个元素是单个Shape的vector）
     * 默认实现：对于单shape算子（get_required_shapes_count() == 1），返回 {1,1}, {10,10}, {100,100}, {1000,1000},
     * {10000,10000} 多shape算子需要重写此方法
     */
    virtual std::vector<std::vector<Shape>> get_default_shapes() const;

    /**
     * @brief 获取所需的shape数量（纯虚函数，由子类实现）
     * @return 所需的shape数量，例如add算子返回1，matmul算子返回2
     */
    virtual size_t get_required_shapes_count() const = 0;

    /**
     * @brief 验证形状是否有效（可选，子类可重写）
     * @param shapes 要验证的形状列表
     * @throws std::exception 如果形状无效
     */
    virtual void validate_shapes(const std::vector<Shape> &shapes) const {}

    /**
     * @brief 获取算子名称（用于帮助信息）
     * @return 算子名称字符串
     */
    virtual std::string get_operator_name() const = 0;

    /**
     * @brief 获取额外的帮助信息（可选，子类可重写）
     * @return 额外的帮助信息字符串，默认为空
     */
    virtual std::string get_additional_help() const { return ""; }

    /**
     * @brief 运行基准测试主函数
     * 解析命令行参数，运行测试循环，输出结果
     * @param argc 命令行参数数量
     * @param argv 命令行参数数组
     * @return 程序退出码（0表示成功）
     */
    int run(int argc, char *argv[]);

private:
    /**
     * @brief 打印使用说明
     * @param program_name 程序名称
     */
    void usage(const char *program_name) const;

    /**
     * @brief 解析命令行参数
     * @param argc 命令行参数数量
     * @param argv 命令行参数数组
     * @param shapes_list 输出：解析后的形状列表，每个元素是一个shape组合
     * @param devices 输出：解析后的设备列表（支持设备索引，如 cuda:0, cuda:1）
     * @param warmup_cnt 输出：预热次数
     * @param repeat_cnt 输出：重复测试次数
     * @return 是否成功解析（false表示需要退出程序，如显示帮助）
     */
    bool parse_arguments(int argc,
                         char *argv[],
                         std::vector<std::vector<Shape>> &shapes_list,
                         std::vector<Device> &devices,
                         int &warmup_cnt,
                         int &repeat_cnt,
                         bool &inplace) const;

    /**
     * @brief 处理设备列表，设置默认设备或验证CUDA可用性
     * @param devices 设备列表（输入输出）
     * @param use_default_devices 是否使用默认设备
     */
    void process_devices(std::vector<Device> &devices, bool use_default_devices) const;
};

}  // namespace benchmark
}  // namespace origin

#endif  // __BENCHMARK_COMMON_BENCHMARK_FRAMEWORK_H__
