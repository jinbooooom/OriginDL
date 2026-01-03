// YOLOv5 推理示例（使用 PNNX 模型）
#include <iostream>
#include <vector>
#include <string>
#include "origin.h"
#include "origin/pnnx/pnnx_graph.h"
#include "origin/utils/log.h"
#include "origin/core/config.h"
#ifdef WITH_CUDA
#include "origin/cuda/cuda.h"
#endif

using namespace origin;
using namespace origin::pnnx;

/**
 * @brief 简化的图像预处理（用于测试）
 * @details 创建一个简单的测试输入张量
 */
Tensor create_test_input(int batch_size = 1, int channels = 3, int height = 640, int width = 640)
{
    // 确定设备类型
    Device device(DeviceType::kCPU);
#ifdef WITH_CUDA
    if (cuda::is_available())
    {
        device = Device(DeviceType::kCUDA, 0);
        logi("CUDA is available. Using GPU for inference.");
    }
    else
    {
        logw("CUDA is not available. Using CPU for inference.");
    }
#else
    logi("CUDA support not compiled. Using CPU for inference.");
#endif
    
    // 创建随机输入数据（实际应用中应该从图像文件加载）
    std::vector<float> input_data(batch_size * channels * height * width);
    for (size_t i = 0; i < input_data.size(); ++i)
    {
        input_data[i] = static_cast<float>(i % 255) / 255.0f;  // 归一化到 [0, 1]
    }
    
    Shape input_shape{static_cast<size_t>(batch_size), static_cast<size_t>(channels),
                      static_cast<size_t>(height), static_cast<size_t>(width)};
    
    // 创建 tensor 并设置设备
    return Tensor(input_data, input_shape, dtype(DataType::kFloat32).device(device));
}

/**
 * @brief YOLOv5 推理示例
 */
void YoloDemo(const std::string &param_path, const std::string &bin_path)
{
    std::cout << "=== YOLOv5 Inference Demo ===" << std::endl;
    std::cout << "Loading model from:" << std::endl;
    std::cout << "  Param: " << param_path << std::endl;
    std::cout << "  Bin: " << bin_path << std::endl;
    
    try
    {
        // 创建 PNNX 图
        PNNXGraph graph(param_path, bin_path);
        
        // 构建计算图
        std::cout << "Building graph..." << std::endl;
        graph.build();
        std::cout << "Graph built successfully!" << std::endl;
        
        // 创建输入
        std::cout << "Creating input tensor..." << std::endl;
        auto input = create_test_input(1, 3, 640, 640);
        std::cout << "Input shape: " << input.shape().to_string() << std::endl;
        
        // 设置输入
        std::cout << "Setting inputs..." << std::endl;
        graph.set_inputs("pnnx_input_0", {input});
        
        // 执行推理
        std::cout << "Running inference..." << std::endl;
        graph.forward(true);  // debug = true 输出调试信息
        
        // 获取输出
        std::cout << "Getting outputs..." << std::endl;
        
        // 尝试直接查找输出节点
        try
        {
            auto outputs = graph.get_outputs("pnnx_output_0");
            
            std::cout << "Outputs size: " << outputs.size() << std::endl;
            
            // 调试：检查所有节点的 output_tensors
            std::cout << "Debug: Checking node outputs..." << std::endl;
            
            if (!outputs.empty())
        {
            logi("Inference successful! Output count: {}", outputs.size());
            std::cout << "Inference successful!" << std::endl;
            std::cout << "Output count: " << outputs.size() << std::endl;
            
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                std::cout << "Output " << i << " shape: " << outputs[i].shape().to_string() << std::endl;
                
                // 打印输出的一些统计信息
                logi("Converting output {} to vector (device: {})", i, outputs[i].device().type() == DeviceType::kCUDA ? "CUDA" : "CPU");
                auto output_data = outputs[i].to_vector<float>();
                logi("Output {} converted to vector, size: {}", i, output_data.size());
                if (!output_data.empty())
                {
                    float min_val = output_data[0];
                    float max_val = output_data[0];
                    float sum = 0.0f;
                    
                    for (float val : output_data)
                    {
                        if (val < min_val) min_val = val;
                        if (val > max_val) max_val = val;
                        sum += val;
                    }
                    
                    float mean = sum / output_data.size();
                    
                    std::cout << "  Min: " << min_val << std::endl;
                    std::cout << "  Max: " << max_val << std::endl;
                    std::cout << "  Mean: " << mean << std::endl;
                    std::cout << "  Size: " << output_data.size() << std::endl;
                }
            }
        }
            else
            {
                std::cout << "Warning: No outputs received!" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error getting outputs: " << e.what() << std::endl;
            throw;
        }
        
        logi("=== YOLOv5 Inference Complete ===");
        std::cout << "=== Inference Complete ===" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        throw;
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <param_path> <bin_path>" << std::endl;
        std::cout << "Example: " << argv[0] << " yolov5s.pnnx.param yolov5s.pnnx.bin" << std::endl;
        return 1;
    }
    
    std::string param_path = argv[1];
    std::string bin_path = argv[2];
    
    try
    {
        YoloDemo(param_path, bin_path);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}

