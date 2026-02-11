// ResNet 推理示例（使用 PNNX 模型）
#include <getopt.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "class_labels.h"
#include "origin.h"

#include <opencv2/opencv.hpp>

using namespace origin;
namespace F = origin::functional;
using namespace origin::pnnx;

/**
 * @brief 图像预处理：Resize + BGR2RGB + 归一化 + 标准化
 * @param image 输入图像（BGR格式）
 * @param device 设备类型
 * @return 预处理后的 Tensor，形状为 (1, 3, 224, 224)
 */
Tensor preprocess_image(const cv::Mat &image, const Device &device)
{
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
    for (int c = 0; c < input_c; ++c)
    {
        const cv::Mat &channel = split_images[c];
        // 转置：从 (H, W) 到 (W, H)
        // KuiperInferGitee 使用列主序，转置后直接 memcpy
        // origindl 使用行主序，需要按行主序存储
        cv::Mat transposed = channel.t();

        // 对于行主序，我们需要按 (H, W) 顺序存储
        // 转置后的 Mat 是 (W, H)，我们需要按行主序存储为 (H, W)
        // 实际上，转置后的数据在内存中的布局是列主序的（OpenCV Mat 默认列主序）
        // 我们需要将其转换为行主序
        for (int h = 0; h < input_h; ++h)
        {
            for (int w = 0; w < input_w; ++w)
            {
                // 转置后的 Mat: (W, H)，访问 transposed(w, h) 得到原始 channel(h, w)
                // 行主序索引：c * H * W + h * W + w
                input_data[c * input_h * input_w + h * input_w + w] = transposed.at<float>(w, h);
            }
        }
    }

    Shape input_shape{1, static_cast<size_t>(input_c), static_cast<size_t>(input_h), static_cast<size_t>(input_w)};

    Tensor result = Tensor(input_data, input_shape, dtype(DataType::kFloat32).device(device));

    // 7. 归一化到 [0, 1]（与 KuiperInferGitee 顺序一致：先转置 memcpy，再归一化）
    auto data_vec = result.to_vector<float>();
    for (size_t i = 0; i < data_vec.size(); ++i)
    {
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
    for (size_t i = 0; i < channel_size; ++i)
    {
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

/**
 * @brief 配置结构体
 */
struct InferenceConfig
{
    std::string image_path;
    std::string param_path;
    std::string bin_path;
    int gpu_id = 0;
};

/**
 * @brief 打印帮助信息
 */
void usage(const char *program_name)
{
    loga("Usage: {} [OPTIONS]", program_name);
    loga("Required options:");
    loga("  -i, --image PATH     Input image file path");
    loga("Optional options:");
    loga("  -p, --param PATH     PNNX param file path (default: auto-detect)");
    loga("  -b, --bin PATH       PNNX bin file path (default: auto-detect)");
    loga("  -g, --gpu INT        GPU device ID (default: 0)");
    loga("  -h, --help           Show this help message");
    loga("Examples:\n");
    loga("  {} -i path/to/image.jpg", program_name);
    loga("  {} -i path/to/image.jpg -p model.pnnx.param -b model.pnnx.bin", program_name);
    loga("  {} -i path/to/image.jpg -g 0", program_name);
}

/**
 * @brief 解析命令行参数
 * @param argc 参数个数
 * @param argv 参数数组
 * @return InferenceConfig 配置对象
 */
InferenceConfig parse_args(int argc, char *argv[])
{
    InferenceConfig config;

    // 定义长选项
    static struct option long_options[] = {{"image", required_argument, 0, 'i'}, {"param", required_argument, 0, 'p'},
                                           {"bin", required_argument, 0, 'b'},   {"gpu", required_argument, 0, 'g'},
                                           {"help", no_argument, 0, 'h'},        {0, 0, 0, 0}};

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "i:p:b:g:h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'i':
                config.image_path = optarg;
                break;
            case 'p':
                config.param_path = optarg;
                break;
            case 'b':
                config.bin_path = optarg;
                break;
            case 'g':
                config.gpu_id = std::atoi(optarg);
                if (config.gpu_id < 0)
                {
                    logw("Invalid GPU ID: {}. Using default: 0", optarg);
                    config.gpu_id = 0;
                }
                break;
            case 'h':
                usage(argv[0]);
                std::exit(0);
            case '?':
                // getopt_long 已经打印了错误信息
                logw("Use -h or --help for usage information");
                break;
            default:
                break;
        }
    }

    // 检查必需参数
    if (config.image_path.empty())
    {
        loge("Error: Image path is required");
        usage(argv[0]);
        std::exit(1);
    }

    return config;
}

/**
 * @brief ResNet 推理示例
 */
int main(int argc, char *argv[])
{
    // 解析命令行参数
    InferenceConfig config = parse_args(argc, argv);

    // 加载图像
    cv::Mat image = cv::imread(config.image_path);
    if (image.empty())
    {
        loge("Error: Cannot load image from {}", config.image_path);
        return -1;
    }

    logi("Image size: {}x{}", image.cols, image.rows);

    // 确定设备
    Device device(DeviceType::kCPU);
    if (cuda::is_available())
    {
        device = Device(DeviceType::kCUDA, config.gpu_id);
        logi("Using CUDA device {}", config.gpu_id);
    }
    else
    {
        logi("CUDA not available, using CPU");
    }

    // 图像预处理
    Tensor input = preprocess_image(image, device);
    logi("Preprocessed input shape: {}", input.shape().to_string());

    // 模型路径处理
    std::string param_path = config.param_path;
    std::string bin_path   = config.bin_path;

    // 如果未指定模型路径，尝试自动检测
    if (param_path.empty() || bin_path.empty())
    {
        // 尝试多个可能的路径
        param_path = "model/pnnx/resnet/resnet18_batch1.pnnx.param";
        bin_path   = "model/pnnx/resnet/resnet18_batch1.pnnx.bin";

        // 如果从 build/bin/example 运行，需要向上三级
        std::ifstream test_file(param_path);
        if (!test_file.good())
        {
            param_path = "../../../model/pnnx/resnet/resnet18_batch1.pnnx.param";
            bin_path   = "../../../model/pnnx/resnet/resnet18_batch1.pnnx.bin";
        }
        test_file.close();
    }

    logi("Using model: param={}, bin={}", param_path, bin_path);

    // 创建 PNNX 图
    PNNXGraph graph(param_path, bin_path);

    // 构建计算图
    logi("Building graph...");
    graph.build();
    logi("Graph built successfully!");

    // 设置输入
    graph.set_inputs("pnnx_input_0", {input});

    // 推理
    logi("Running inference...");
    auto start_time = std::chrono::high_resolution_clock::now();
    graph.forward(false);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    logi("Forward time: {:.3f}s", duration.count() / 1000.0);

    // 获取输出
    std::vector<Tensor> outputs = graph.get_outputs("pnnx_output_0");
    if (outputs.empty())
    {
        loge("Error: No output found");
        return -1;
    }

    Tensor output = outputs[0];
    logi("Output shape: {}", output.shape().to_string());

    // 应用 Softmax
    // ResNet 输出形状应该是 (1, 1000)，对最后一个维度应用 softmax
    Tensor softmax_output = F::softmax(output, -1);

    // 找到最大概率的类别
    auto prob_data = softmax_output.to_vector<float>();
    float max_prob = -1.0f;
    int max_index  = -1;

    for (size_t i = 0; i < prob_data.size(); ++i)
    {
        if (prob_data[i] > max_prob)
        {
            max_prob  = prob_data[i];
            max_index = static_cast<int>(i);
        }
    }

    // 显示类别名称（如果可用）
    std::string class_name = "Unknown";
    if (max_index >= 0 && static_cast<size_t>(max_index) < IMAGENET_CLASSES.size())
    {
        class_name = IMAGENET_CLASSES[max_index];
    }

    loga("Class with max probability: {} (index: {}, probability: {:.4f})", class_name, max_index, max_prob);

    return 0;
}
