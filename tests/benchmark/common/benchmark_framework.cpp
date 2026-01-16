#include "benchmark/common/benchmark_framework.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <getopt.h>
#include "origin/cuda/cuda.h"

namespace origin {
namespace benchmark {

std::string BenchmarkFramework::device_type_to_string(DeviceType device_type)
{
    return (device_type == DeviceType::kCPU) ? "cpu" : "cuda";
}

std::vector<std::vector<Shape>> BenchmarkFramework::get_default_shapes() const
{
    // 对于单shape算子，提供默认实现
    if (get_required_shapes_count() == 1)
    {
        return {
            {Shape({1, 1})},
            {Shape({10, 10})},
            {Shape({100, 100})},
            {Shape({1000, 1000})},
            {Shape({10000, 10000})},
        };
    }
    // 对于多shape算子，子类必须重写此方法
    THROW_RUNTIME_ERROR("get_default_shapes() must be overridden for operators requiring {} shapes", get_required_shapes_count());
}

void BenchmarkFramework::usage(const char* program_name) const
{
    loga("Usage: {} [OPTIONS]", program_name);
    loga("Options:");
    loga("  -d, --device DEVICE       Device type: cpu or cuda (can be specified multiple times)");
    loga("                            If not specified, tests all available devices");
    loga("  -s, --shape SHAPE         Tensor shape, e.g., \"100,100\" or \"1000,1000\"");
    loga("                            For operators requiring multiple shapes, use ':' to separate");
    loga("                            e.g., \"100,200:200,50\" for matmul");
    loga("                            (can be specified multiple times)");
    loga("                            If not specified, uses default shapes");
    std::string additional_help = get_additional_help();
    if (!additional_help.empty())
    {
        loga("{}", additional_help);
    }
    loga("  -w, --warmup ITERATIONS   Number of warmup iterations (default: 5)");
    loga("  -r, --repeat ITERATIONS   Number of repeat iterations (default: 100)");
    loga("  -h, --help                Show this help message");
}

bool BenchmarkFramework::parse_arguments(int argc, char* argv[],
                                         std::vector<std::vector<Shape>>& shapes_list,
                                         std::vector<DeviceType>& devices,
                                         int& warmup_cnt,
                                         int& repeat_cnt) const
{
    bool use_default_shapes = true;
    bool use_default_devices = true;
    warmup_cnt = 5;
    repeat_cnt = 100;
    size_t required_shapes_count = get_required_shapes_count();

    // 解析命令行参数
    static struct option long_options[] = {
        {"device", required_argument, 0, 'd'},
        {"shape", required_argument, 0, 's'},
        {"warmup", required_argument, 0, 'w'},
        {"repeat", required_argument, 0, 'r'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "d:s:w:r:h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'd':
            {
                std::string device_str = optarg;
                if (device_str == "cpu")
                {
                    if (use_default_devices)
                    {
                        devices.clear();
                        use_default_devices = false;
                    }
                    devices.push_back(DeviceType::kCPU);
                }
                else if (device_str == "cuda")
                {
                    if (use_default_devices)
                    {
                        devices.clear();
                        use_default_devices = false;
                    }
                    devices.push_back(DeviceType::kCUDA);
                }
                else
                {
                    loge("Error: Invalid device '{}'. Use 'cpu' or 'cuda'.", device_str);
                    return false;
                }
                break;
            }
            case 's':
            {
                std::string shape_str = optarg;
                std::vector<Shape> shapes;
                
                // 检查是否包含冒号（多个shape）
                if (shape_str.find(':') != std::string::npos)
                {
                    // 解析多个shape
                    shapes = parse_multiple_shapes_string(shape_str);
                    if (shapes.empty())
                    {
                        loge("Error: Invalid shapes '{}'. Expected format: 'dim1,dim2,...:dim1,dim2,...'", shape_str);
                        return false;
                    }
                }
                else
                {
                    // 解析单个shape
                    Shape shape = parse_shape_string(shape_str);
                    if (shape.ndims() == 0)
                    {
                        loge("Error: Invalid shape '{}'. Expected format: 'dim1,dim2,...'", shape_str);
                        return false;
                    }
                    shapes.push_back(shape);
                }
                
                // 验证shape数量
                if (shapes.size() != required_shapes_count)
                {
                    loge("Error: Operator requires {} shape(s), but got {} shape(s) in '{}'",
                         required_shapes_count, shapes.size(), shape_str);
                    return false;
                }
                
                // 验证形状（子类可以重写此方法）
                try
                {
                    validate_shapes(shapes);
                }
                catch (const std::exception& e)
                {
                    loge("Error: Invalid shapes '{}': {}", shape_str, e.what());
                    return false;
                }
                
                if (use_default_shapes)
                {
                    shapes_list.clear();
                    use_default_shapes = false;
                }
                shapes_list.push_back(shapes);
                break;
            }
            case 'w':
            {
                try
                {
                    warmup_cnt = std::stoi(optarg);
                    if (warmup_cnt < 0)
                    {
                        loge("Error: Warmup count must be non-negative");
                        return false;
                    }
                }
                catch (const std::exception&)
                {
                    loge("Error: Invalid warmup count '{}'", optarg);
                    return false;
                }
                break;
            }
            case 'r':
            {
                try
                {
                    repeat_cnt = std::stoi(optarg);
                    if (repeat_cnt <= 0)
                    {
                        loge("Error: Repeat count must be positive");
                        return false;
                    }
                }
                catch (const std::exception&)
                {
                    loge("Error: Invalid repeat count '{}'", optarg);
                    return false;
                }
                break;
            }
            case 'h':
                usage(argv[0]);
                return false;  // 返回false表示需要退出
            case '?':
                loga("Use -h or --help for usage information");
                return false;
            default:
                break;
        }
    }

    // 如果使用默认shapes，则使用默认列表
    if (use_default_shapes)
    {
        shapes_list = get_default_shapes();
    }

    // 处理设备列表
    process_devices(devices, use_default_devices);

    if (shapes_list.empty())
    {
        loge("Error: No shapes specified");
        return false;
    }

    if (devices.empty())
    {
        loge("Error: No devices specified");
        return false;
    }

    return true;
}

void BenchmarkFramework::process_devices(std::vector<DeviceType>& devices, bool use_default_devices) const
{
    if (use_default_devices)
    {
        devices.push_back(DeviceType::kCPU);
        if (cuda::is_available())
        {
            devices.push_back(DeviceType::kCUDA);
        }
    }
    else
    {
        // 检查指定的CUDA设备是否可用
        bool has_cuda = false;
        for (auto device : devices)
        {
            if (device == DeviceType::kCUDA)
            {
                has_cuda = true;
                break;
            }
        }

        if (has_cuda)
        {
            if (!cuda::is_available())
            {
                logw("Warning: CUDA is not available, skipping CUDA tests");
                devices.erase(std::remove(devices.begin(), devices.end(), DeviceType::kCUDA), devices.end());
            }
        }
    }
}

int BenchmarkFramework::run(int argc, char* argv[])
{
    std::vector<std::vector<Shape>> shapes_list;
    std::vector<DeviceType> devices;
    int warmup_cnt;
    int repeat_cnt;

    // 解析命令行参数
    if (!parse_arguments(argc, argv, shapes_list, devices, warmup_cnt, repeat_cnt))
    {
        // parse_arguments 返回 false 表示需要退出（如显示帮助）
        return 0;
    }

    std::vector<DataType> dtypes = {DataType::kFloat32};

    // 先收集所有结果和字符串，用于计算列宽
    struct ResultRow {
        std::string shape_str;
        std::string repeat_str;
        std::string device_str;
        std::string dtype_str;
        std::string time_str;
        bool valid;
    };
    std::vector<ResultRow> results;

    // 运行测试并收集结果
    for (const auto& shapes : shapes_list)
    {
        for (const auto& dtype : dtypes)
        {
            for (const auto& device_type : devices)
            {
                ResultRow row;
                row.valid = false;

                // 构造shape字符串
                std::string shape_str;
                for (size_t i = 0; i < shapes.size(); ++i)
                {
                    if (i > 0)
                    {
                        shape_str += ":";
                    }
                    shape_str += shapes[i].to_string();
                }
                row.shape_str = shape_str;

                try
                {
                    BenchmarkConfig config{
                        shapes,
                        dtype,
                        Device(device_type, 0),
                        warmup_cnt,
                        repeat_cnt
                    };

                    double avg_time_us = run_benchmark(config);

                    row.repeat_str = std::to_string(repeat_cnt);
                    row.device_str = device_type_to_string(device_type);
                    row.dtype_str = origin::dtype_to_string(dtype);
                    
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(4) << avg_time_us;
                    row.time_str = oss.str();
                    row.valid = true;
                }
                catch (const std::exception& e)
                {
                    loge("Error testing {} {} {}: {}",
                         shape_str,
                         device_type_to_string(device_type),
                         origin::dtype_to_string(dtype),
                         e.what());
                    // row.valid 保持为 false，跳过这一行
                }

                if (row.valid)
                {
                    results.push_back(row);
                }
            }
        }
    }

    // 计算每列的最大宽度
    size_t max_shape_width = 5;  // "shape" 长度
    size_t max_repeat_width = 6; // "repeat" 长度
    size_t max_device_width = 6; // "device" 长度
    size_t max_dtype_width = 5;  // "dtype" 长度
    size_t max_time_width = 17;  // "origindl_time_us" 长度

    for (const auto& row : results)
    {
        max_shape_width = std::max(max_shape_width, row.shape_str.length());
        max_repeat_width = std::max(max_repeat_width, row.repeat_str.length());
        max_device_width = std::max(max_device_width, row.device_str.length());
        max_dtype_width = std::max(max_dtype_width, row.dtype_str.length());
        max_time_width = std::max(max_time_width, row.time_str.length());
    }

    // 输出表头（制表符分隔，方便Python解析，列宽增加3个字符以增加间距）
    // 注意：制表符会在每个tab stop对齐，为了确保间距足够，我们在列宽基础上增加3
    const int extra_spacing = 3;
    std::cout << std::left
              << std::setw(static_cast<int>(max_shape_width + extra_spacing)) << "shape" << "\t"
              << std::setw(static_cast<int>(max_repeat_width + extra_spacing)) << "repeat" << "\t"
              << std::setw(static_cast<int>(max_device_width + extra_spacing)) << "device" << "\t"
              << std::setw(static_cast<int>(max_dtype_width + extra_spacing)) << "dtype" << "\t"
              << std::setw(static_cast<int>(max_time_width + extra_spacing)) << "origindl_time_us"
              << std::endl;

    // 输出结果（使用制表符分隔，方便Python解析）
    for (const auto& row : results)
    {
        std::cout << std::left
                  << std::setw(static_cast<int>(max_shape_width + extra_spacing)) << row.shape_str << "\t"
                  << std::setw(static_cast<int>(max_repeat_width + extra_spacing)) << row.repeat_str << "\t"
                  << std::setw(static_cast<int>(max_device_width + extra_spacing)) << row.device_str << "\t"
                  << std::setw(static_cast<int>(max_dtype_width + extra_spacing)) << row.dtype_str << "\t"
                  << std::setw(static_cast<int>(max_time_width + extra_spacing)) << row.time_str
                  << std::endl;
    }

    return 0;
}

}  // namespace benchmark
}  // namespace origin
