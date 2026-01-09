// ResNet 推理示例（使用 PNNX 模型）
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include "origin.h"
#include "origin/pnnx/pnnx_graph.h"
#include "origin/utils/log.h"
#include "origin/core/config.h"
#include "origin/core/operator.h"
#include "class_labels.h"
#ifdef WITH_CUDA
#include "origin/cuda/cuda.h"
#endif

#ifdef OPENCV_FOUND
#include <opencv2/opencv.hpp>
#endif

using namespace origin;
namespace F = origin::functional;
using namespace origin::pnnx;

#ifdef OPENCV_FOUND
/**
 * @brief 图像预处理：Resize + BGR2RGB + 归一化 + 标准化
 * @param image 输入图像（BGR格式）
 * @param device 设备类型
 * @return 预处理后的 Tensor，形状为 (1, 3, 224, 224)
 */
Tensor preprocess_image(const cv::Mat& image, const Device& device) {
    const int input_h = 224;
    const int input_w = 224;
    const int input_c = 3;
    
    // 1. Resize 到 224x224
    cv::Mat resize_image;
    cv::resize(image, resize_image, cv::Size(input_w, input_h));
    
    // 2. BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(resize_image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 3. 转换为 float（先不归一化，与 KuiperInferGitee 保持一致）
    cv::Mat float_image;
    rgb_image.convertTo(float_image, CV_32FC3);
    
    // 4. 分离通道
    std::vector<cv::Mat> split_images;
    cv::split(float_image, split_images);
    
    // 5. 创建 Tensor 数据：形状为 (1, 3, 224, 224)
    // 注意：origindl 使用行主序，KuiperInferGitee 使用列主序
    // 对于行主序：(N, C, H, W)，索引为 n*C*H*W + c*H*W + h*W + w
    // 对于列主序：(C, H, W)，索引为 c + h*C + w*C*H
    // 为了匹配 KuiperInferGitee 的行为，我们需要正确转换数据布局
    std::vector<float> input_data(input_c * input_h * input_w);
    
    // 6. 将 OpenCV Mat 数据转换为 Tensor 格式（行主序）
    // Tensor 格式：NCHW (Batch, Channel, Height, Width) - 行主序
    // OpenCV Mat 格式：HWC (Height, Width, Channel)
    // KuiperInferGitee: 转置后 memcpy，然后归一化
    // origindl: 需要匹配相同的数据布局
    for (int c = 0; c < input_c; ++c) {
        const cv::Mat& channel = split_images[c];
        // 转置：从 (H, W) 到 (W, H)
        // KuiperInferGitee 使用列主序，转置后直接 memcpy
        // origindl 使用行主序，需要按行主序存储
        cv::Mat transposed = channel.t();
        
        // 对于行主序，我们需要按 (H, W) 顺序存储
        // 转置后的 Mat 是 (W, H)，我们需要按行主序存储为 (H, W)
        // 实际上，转置后的数据在内存中的布局是列主序的（OpenCV Mat 默认列主序）
        // 我们需要将其转换为行主序
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                // 转置后的 Mat: (W, H)，访问 transposed(w, h) 得到原始 channel(h, w)
                // 行主序索引：c * H * W + h * W + w
                input_data[c * input_h * input_w + h * input_w + w] = transposed.at<float>(w, h);
            }
        }
    }
    
    Shape input_shape{1, static_cast<size_t>(input_c), 
                     static_cast<size_t>(input_h), static_cast<size_t>(input_w)};
    
    Tensor result = Tensor(input_data, input_shape, dtype(DataType::kFloat32).device(device));
    
    // 7. 归一化到 [0, 1]（与 KuiperInferGitee 顺序一致：先转置 memcpy，再归一化）
    auto data_vec = result.to_vector<float>();
    for (size_t i = 0; i < data_vec.size(); ++i) {
        data_vec[i] = data_vec[i] / 255.0f;
    }
    result = Tensor(data_vec, input_shape, dtype(DataType::kFloat32).device(device));
    
    // 8. 标准化：减去均值并除以标准差
    // ImageNet 标准化参数
    float mean_r = 0.485f;
    float mean_g = 0.456f;
    float mean_b = 0.406f;
    
    float std_r = 0.229f;
    float std_g = 0.224f;
    float std_b = 0.225f;
    
    // 获取数据
    data_vec = result.to_vector<float>();
    
    // 对每个通道进行标准化
    size_t channel_size = input_h * input_w;
    for (size_t i = 0; i < channel_size; ++i) {
        // R 通道
        data_vec[i] = (data_vec[i] - mean_r) / std_r;
        // G 通道
        data_vec[i + channel_size] = (data_vec[i + channel_size] - mean_g) / std_g;
        // B 通道
        data_vec[i + 2 * channel_size] = (data_vec[i + 2 * channel_size] - mean_b) / std_b;
    }
    
    // 创建标准化后的 Tensor
    return Tensor(data_vec, input_shape, dtype(DataType::kFloat32).device(device));
}
#endif  // OPENCV_FOUND

/**
 * @brief ResNet 推理示例
 */
int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    
    const std::string image_path = argv[1];
    
#ifdef OPENCV_FOUND
    // 加载图像
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Cannot load image from " << image_path << std::endl;
        return -1;
    }
    
    std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
    
    // 确定设备
    Device device(DeviceType::kCPU);
#ifdef WITH_CUDA
    if (cuda::is_available()) {
        device = Device(DeviceType::kCUDA, 0);
        std::cout << "Using CUDA device" << std::endl;
    } else {
        std::cout << "CUDA not available, using CPU" << std::endl;
    }
#else
    std::cout << "Using CPU" << std::endl;
#endif
    
    // 图像预处理
    Tensor input = preprocess_image(image, device);
    std::cout << "Preprocessed input shape: " << input.shape().to_string() << std::endl;
    
    // 模型路径（从项目根目录运行，或从 build/bin/example 运行时使用相对路径）
    // 尝试多个可能的路径
    std::string param_path = "s/KuiperInferGitee/tmp/resnet/demo/resnet18_batch1.pnnx.param";
    std::string bin_path = "s/KuiperInferGitee/tmp/resnet/demo/resnet18_batch1.pnnx.bin";
    
    // 如果从 build/bin/example 运行，需要向上三级
    std::ifstream test_file(param_path);
    if (!test_file.good()) {
        param_path = "../../../s/KuiperInferGitee/tmp/resnet/demo/resnet18_batch1.pnnx.param";
        bin_path = "../../../s/KuiperInferGitee/tmp/resnet/demo/resnet18_batch1.pnnx.bin";
    }
    test_file.close();
    
    // 创建 PNNX 图
    PNNXGraph graph(param_path, bin_path);
    
    // 构建计算图
    std::cout << "Building graph..." << std::endl;
    graph.build();
    std::cout << "Graph built successfully!" << std::endl;
    
    // 设置输入
    graph.set_inputs("pnnx_input_0", {input});
    
    // 推理
    std::cout << "Running inference..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    graph.forward(false);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "Forward time: " << duration.count() / 1000.0 << "s" << std::endl;
    
    // 获取输出
    std::vector<Tensor> outputs = graph.get_outputs("pnnx_output_0");
    if (outputs.empty()) {
        std::cerr << "Error: No output found" << std::endl;
        return -1;
    }
    
    Tensor output = outputs[0];
    std::cout << "Output shape: " << output.shape().to_string() << std::endl;
    
    // 应用 Softmax
    // ResNet 输出形状应该是 (1, 1000)，对最后一个维度应用 softmax
    Tensor softmax_output = F::softmax(output, -1);
    
    // 找到最大概率的类别
    auto prob_data = softmax_output.to_vector<float>();
    float max_prob = -1.0f;
    int max_index = -1;
    
    for (size_t i = 0; i < prob_data.size(); ++i) {
        if (prob_data[i] > max_prob) {
            max_prob = prob_data[i];
            max_index = static_cast<int>(i);
        }
    }
    
    // 显示类别名称（如果可用）
    std::string class_name = "Unknown";
    if (max_index >= 0 && static_cast<size_t>(max_index) < IMAGENET_CLASSES.size()) {
        class_name = IMAGENET_CLASSES[max_index];
    }
    
    std::cout << "Class with max probability: " << class_name 
              << " (index: " << max_index << ", probability: " << max_prob << ")" << std::endl;
    
    return 0;
#else
    std::cerr << "Error: OpenCV not found. Please install OpenCV to use this demo." << std::endl;
    return -1;
#endif  // OPENCV_FOUND
}

