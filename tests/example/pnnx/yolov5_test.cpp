// YOLOv5 推理示例（使用 PNNX 模型）
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <getopt.h>
#include "origin.h"
#include "origin/pnnx/pnnx_graph.h"
#include "origin/utils/log.h"
#include "origin/core/config.h"
#ifdef WITH_CUDA
#include "origin/cuda/cuda.h"
#endif

#ifdef OPENCV_FOUND
#include <opencv2/opencv.hpp>
#endif

using namespace origin;
using namespace origin::pnnx;

/**
 * @brief 用户配置结构体
 */
struct UserCfg {
    std::string param_path;         // PNNX param 文件路径
    std::string bin_path;            // PNNX bin 文件路径
    std::string image_path;          // 输入图像路径
    std::string output_path;         // 输出图像路径
    float confidence_thresh = 0.25f; // 置信度阈值（与 KuiperInferGitee 保持一致）
    float iou_thresh = 0.25f;        // IOU 阈值（与 KuiperInferGitee 保持一致）
    int gpu_device = 0;              // GPU 设备 ID（默认使用 gpu0）
    int batch_size = 1;              // Batch size（默认 1，对于 batch4 模型应设置为 4）
    int input_h = 640;               // 输入图像高度（默认 640，会从 param 文件自动解析）
    int input_w = 640;               // 输入图像宽度（默认 640，会从 param 文件自动解析）
    bool debug = false;              // 是否输出调试信息
    bool show_help = false;          // 是否显示帮助信息
};

/**
 * @brief 从 PNNX param 文件中解析输入形状信息
 * @param param_path param 文件路径
 * @param batch_size 输出的 batch_size
 * @param input_h 输出的 input height
 * @param input_w 输出的 input width
 * @return 是否解析成功
 */
bool GetInputShapeFromParamFile(const std::string& param_path, 
                                uint32_t& batch_size, 
                                int32_t& input_h, 
                                int32_t& input_w) {
    std::ifstream file(param_path);
    if (!file.is_open()) {
        std::cerr << "Warning: Failed to open param file: " << param_path 
                  << ", using default values" << std::endl;
        batch_size = 1;
        input_h = 640;
        input_w = 640;
        return false;
    }

    std::string line;
    // 跳过 magic number (第1行)
    std::getline(file, line);
    // 跳过 operator count (第2行)
    std::getline(file, line);
    
    // 读取第一个输入节点 (第3行)
    if (std::getline(file, line)) {
        // 查找 #0=( 的位置
        size_t shape_start = line.find("#0=(");
        if (shape_start != std::string::npos) {
            // 找到形状字符串的起始位置
            size_t shape_begin = shape_start + 4; // "#0=(" 的长度
            size_t shape_end = line.find(')', shape_begin);
            
            if (shape_end != std::string::npos) {
                // 提取形状字符串，例如 "4,3,320,320"
                std::string shape_str = line.substr(shape_begin, shape_end - shape_begin);
                
                // 解析所有维度：batch, channels, height, width
                std::istringstream iss(shape_str);
                std::vector<std::string> dims;
                std::string dim;
                while (std::getline(iss, dim, ',')) {
                    dims.push_back(dim);
                }
                
                if (dims.size() >= 4) {
                    try {
                        batch_size = std::stoul(dims[0]);
                        // dims[1] 是 channels，不需要
                        input_h = std::stoi(dims[2]);
                        input_w = std::stoi(dims[3]);
                        std::cout << "Parsed input shape from param file: batch_size=" << batch_size 
                                  << ", input_h=" << input_h << ", input_w=" << input_w << std::endl;
                        return true;
                    } catch (const std::exception& e) {
                        std::cerr << "Warning: Failed to parse shape dimensions, error: " << e.what() 
                                  << ", using default values" << std::endl;
                    }
                }
            }
        }
    }
    
    std::cerr << "Warning: Failed to parse input shape from param file: " << param_path 
              << ", using default values" << std::endl;
    batch_size = 1;
    input_h = 640;
    input_w = 640;
    return false;
}

/**
 * @brief 解析命令行参数
 */
UserCfg parse_args(int argc, char *argv[]) {
    UserCfg cfg;
    
    static struct option long_options[] = {
        {"param",      required_argument, 0, 'p'},
        {"bin",        required_argument, 0, 'b'},
        {"image",      required_argument, 0, 'i'},
        {"output",     required_argument, 0, 'o'},
        {"confidence", required_argument, 0, 'c'},
        {"iou",        required_argument, 0, 'u'},
        {"gpu",        required_argument, 0, 'g'},
        {"batch",      required_argument, 0, 'B'},
        {"debug",      no_argument,       0, 'd'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "p:b:i:o:c:u:g:B:dh", long_options, &option_index)) != -1) {
        switch (c) {
            case 'p':
                cfg.param_path = optarg;
                break;
            case 'b':
                cfg.bin_path = optarg;
                break;
            case 'i':
                cfg.image_path = optarg;
                break;
            case 'o':
                cfg.output_path = optarg;
                break;
            case 'c':
                cfg.confidence_thresh = std::stof(optarg);
                break;
            case 'u':
                cfg.iou_thresh = std::stof(optarg);
                break;
            case 'g':
                cfg.gpu_device = std::stoi(optarg);
                break;
            case 'B':
                cfg.batch_size = std::stoi(optarg);
                break;
            case 'd':
                cfg.debug = true;
                break;
            case 'h':
                cfg.show_help = true;
                break;
            case '?':
                // getopt_long 已经打印了错误信息
                exit(1);
            default:
                break;
        }
    }
    
    // 兼容旧的参数格式：位置参数
    if (cfg.param_path.empty() && optind < argc) {
        cfg.param_path = argv[optind++];
    }
    if (cfg.bin_path.empty() && optind < argc) {
        cfg.bin_path = argv[optind++];
    }
    if (cfg.image_path.empty() && optind < argc) {
        cfg.image_path = argv[optind++];
    }
    
    return cfg;
}

/**
 * @brief 打印帮助信息
 */
void print_help(const char *program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n"
              << "\n"
              << "Required options:\n"
              << "  -p, --param PATH      PNNX param file path\n"
              << "  -b, --bin PATH        PNNX bin file path\n"
              << "\n"
              << "Optional options:\n"
              << "  -i, --image PATH      Input image path (default: use test input)\n"
              << "  -o, --output PATH     Output image path (default: output_detection.jpg)\n"
              << "  -c, --confidence FLOAT Confidence threshold (default: 0.25)\n"
              << "  -u, --iou FLOAT       IOU threshold for NMS (default: 0.45)\n"
              << "  -g, --gpu INT         GPU device ID (default: 0)\n"
              << "  -B, --batch INT       Batch size (default: 1, use 4 for batch4 models)\n"
              << "  -d, --debug           Enable debug logging\n"
              << "  -h, --help            Show this help message\n"
              << "\n"
              << "Examples:\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i image.jpg\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i image.jpg -c 0.5 -u 0.5\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i image.jpg -B 4  (for batch4 models)\n"
              << "  " << program_name << " model.pnnx.param model.pnnx.bin image.jpg  (legacy format)\n"
              << std::endl;
}

#ifdef OPENCV_FOUND
// 检测结果结构
struct Detection {
    cv::Rect box;
    float conf = 0.f;
    int class_id = -1;
};

/**
 * @brief Letterbox 图像预处理
 * @param image 输入图像
 * @param out_image 输出图像
 * @param new_shape 目标尺寸
 * @param stride 步长
 * @param color 填充颜色
 * @param fixed_shape 是否固定形状
 * @param scale_up 是否放大
 * @return 缩放比例
 */
float Letterbox(const cv::Mat& image, cv::Mat& out_image, 
                const cv::Size& new_shape = cv::Size(640, 640), 
                int stride = 32,
                const cv::Scalar& color = cv::Scalar(114, 114, 114), 
                bool fixed_shape = false,
                bool scale_up = false) {
    cv::Size shape = image.size();
    float r = std::min((float)new_shape.height / (float)shape.height,
                       (float)new_shape.width / (float)shape.width);
    if (!scale_up) {
        r = std::min(r, 1.0f);
    }

    int new_unpad[2]{(int)std::round((float)shape.width * r),
                     (int)std::round((float)shape.height * r)};

    cv::Mat tmp;
    if (shape.width != new_unpad[0] || shape.height != new_unpad[1]) {
        cv::resize(image, tmp, cv::Size(new_unpad[0], new_unpad[1]));
    } else {
        tmp = image.clone();
    }

    float dw = new_shape.width - new_unpad[0];
    float dh = new_shape.height - new_unpad[1];

    if (!fixed_shape) {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }

    dw /= 2.0f;
    dh /= 2.0f;

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    return 1.0f / r;
}

/**
 * @brief 裁剪坐标到有效范围
 */
template <typename T>
T clip(const T& n, const T& lower, const T& upper) {
    return std::max(lower, std::min(n, upper));
}

/**
 * @brief 将坐标从模型输入尺寸转换回原始图像尺寸
 */
void ScaleCoords(const cv::Size& img_shape, cv::Rect& coords, const cv::Size& img_origin_shape) {
    float gain = std::min((float)img_shape.height / (float)img_origin_shape.height,
                          (float)img_shape.width / (float)img_origin_shape.width);

    int pad[2] = {(int)(((float)img_shape.width - (float)img_origin_shape.width * gain) / 2.0f),
                  (int)(((float)img_shape.height - (float)img_origin_shape.height * gain) / 2.0f)};

    coords.x = (int)std::round(((float)(coords.x - pad[0]) / gain));
    coords.y = (int)std::round(((float)(coords.y - pad[1]) / gain));

    coords.width = (int)std::round(((float)coords.width / gain));
    coords.height = (int)std::round(((float)coords.height / gain));

    coords.x = clip(coords.x, 0, img_origin_shape.width);
    coords.y = clip(coords.y, 0, img_origin_shape.height);
    coords.width = clip(coords.width, 0, img_origin_shape.width);
    coords.height = clip(coords.height, 0, img_origin_shape.height);
}

/**
 * @brief 图像预处理：Letterbox + 归一化 + BGR2RGB + 转换为 Tensor
 */
Tensor preprocess_image(const cv::Mat& image, const Device& device, int input_h = 640, int input_w = 640) {
    const int input_c = 3;
    
    // Letterbox
    cv::Mat out_image;
    Letterbox(image, out_image, {input_h, input_w}, 32, {114, 114, 114}, true);
    
    // BGR to RGB
    cv::Mat rgb_image;
    cv::cvtColor(out_image, rgb_image, cv::COLOR_BGR2RGB);
    
    // 归一化到 [0, 1]
    cv::Mat normalize_image;
    rgb_image.convertTo(normalize_image, CV_32FC3, 1.0 / 255.0);
    
    // 分离通道
    std::vector<cv::Mat> split_images;
    cv::split(normalize_image, split_images);
    
    // 创建 Tensor 数据：形状为 (1, 3, H, W)
    std::vector<float> input_data(input_c * input_h * input_w);
    
    // 将 OpenCV Mat 数据转换为 Tensor 格式
    // Tensor 格式：NCHW (Batch, Channel, Height, Width) - 行主序
    // OpenCV Mat 格式：HWC (Height, Width, Channel)
    // KuiperInferGitee 使用列主序，转置后直接 memcpy
    // origindl 使用行主序，需要按行主序存储（与 ResNet 预处理保持一致）
    for (int c = 0; c < input_c; ++c) {
        const cv::Mat& channel = split_images[c];
        // 转置：从 (H, W) 到 (W, H)
        // KuiperInferGitee 使用列主序，转置后直接 memcpy
        // origindl 使用行主序，需要按行主序存储
        cv::Mat transposed = channel.t();
        
        // 对于行主序，我们需要按 (H, W) 顺序存储
        // 转置后的 Mat 是 (W, H)，我们需要按行主序存储为 (H, W)
        // 行主序索引：c * H * W + h * W + w
        for (int h = 0; h < input_h; ++h) {
            for (int w = 0; w < input_w; ++w) {
                // 转置后的 Mat: (W, H)，访问 transposed(w, h) 得到原始 channel(h, w)
                input_data[c * input_h * input_w + h * input_w + w] = transposed.at<float>(w, h);
            }
        }
    }
    
    Shape input_shape{1, static_cast<size_t>(input_c), 
                     static_cast<size_t>(input_h), static_cast<size_t>(input_w)};
    
    Tensor result = Tensor(input_data, input_shape, dtype(DataType::kFloat32).device(device));
    
    // 调试：输出预处理后的数据统计（始终输出，用于对比）
    {
        auto data_vec = result.to_vector<float>();
        float min_val = *std::min_element(data_vec.begin(), data_vec.end());
        float max_val = *std::max_element(data_vec.begin(), data_vec.end());
        float mean_val = std::accumulate(data_vec.begin(), data_vec.end(), 0.0f) / data_vec.size();
        std::cout << "\n=== DEBUG: origindl Preprocessed Input ===" << std::endl;
        std::cout << "Input shape: " << result.shape().to_string() << std::endl;
        std::cout << "Input data stats: min=" << min_val << ", max=" << max_val 
                  << ", mean=" << mean_val << std::endl;
        std::cout << "Input first 10 values: ";
        for (size_t i = 0; i < std::min(size_t(10), data_vec.size()); ++i) {
            std::cout << data_vec[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "===================================\n" << std::endl;
    }
    
    return result;
}
#endif  // OPENCV_FOUND

/**
 * @brief 简化的图像预处理（用于测试，当没有 OpenCV 时）
 */
Tensor create_test_input(const Device& device, int batch_size = 1, int channels = 3, int height = 640, int width = 640)
{
    
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
void YoloDemo(const UserCfg &cfg)
{
    std::cout << "=== YOLOv5 Inference Demo ===" << std::endl;
    std::cout << "Loading model from:" << std::endl;
    std::cout << "  Param: " << cfg.param_path << std::endl;
    std::cout << "  Bin: " << cfg.bin_path << std::endl;
    
    // 检测并选择设备（GPU优先，如果没有GPU则使用CPU）
    // 注意：cuDNN 和 cuBLAS 已在代码中禁用，将使用自定义 GPU kernel（行主序）
    Device device(DeviceType::kCPU);
#ifdef WITH_CUDA
    if (cuda::is_available() && cfg.gpu_device >= 0)
    {
        device = Device(DeviceType::kCUDA, cfg.gpu_device);
        logi("CUDA is available. Using GPU{} for inference (cuDNN/cuBLAS disabled, using custom kernels).", cfg.gpu_device);
    }
    else
    {
        logw("CUDA is not available. Using CPU for inference.");
    }
#else
    logi("CUDA support not compiled. Using CPU for inference.");
#endif
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Confidence threshold: " << cfg.confidence_thresh << std::endl;
    std::cout << "  IOU threshold: " << cfg.iou_thresh << std::endl;
    std::cout << "  Device: " << device.to_string() << std::endl;
    if (!cfg.image_path.empty()) {
        std::cout << "  Input image: " << cfg.image_path << std::endl;
    }
    
    try
    {
        // 创建 PNNX 图
        PNNXGraph graph(cfg.param_path, cfg.bin_path);
        
        // 构建计算图
        std::cout << "Building graph..." << std::endl;
        graph.build();
        std::cout << "Graph built successfully!" << std::endl;
        
        // 创建输入（支持 batch size）
        Tensor input;
#ifdef OPENCV_FOUND
        if (!cfg.image_path.empty()) {
            std::cout << "Loading image: " << cfg.image_path << std::endl;
            cv::Mat image = cv::imread(cfg.image_path);
            if (image.empty()) {
                std::cerr << "Error: Cannot load image from " << cfg.image_path << std::endl;
                return;
            }
            std::cout << "Image size: " << image.cols << "x" << image.rows << std::endl;
            
            // 创建单个输入，然后复制到 batch（使用解析出的输入尺寸）
            Tensor single_input = preprocess_image(image, device, cfg.input_h, cfg.input_w);
            if (cfg.batch_size == 1) {
                input = single_input;
            } else {
                // 将单个输入复制 batch_size 次
                Shape single_shape = single_input.shape();
                size_t single_size = single_shape.elements();
                std::vector<float> batch_data(cfg.batch_size * single_size);
                auto single_data = single_input.to_vector<float>();
                for (int i = 0; i < cfg.batch_size; ++i) {
                    std::copy(single_data.begin(), single_data.end(), 
                             batch_data.begin() + i * single_size);
                }
                std::vector<size_t> batch_dims = single_shape.dims();
                batch_dims[0] = static_cast<size_t>(cfg.batch_size);
                input = Tensor(batch_data, Shape(batch_dims), 
                              dtype(DataType::kFloat32).device(device));
            }
        } else {
            std::cout << "No image path provided, using test input..." << std::endl;
            input = create_test_input(device, cfg.batch_size, 3, cfg.input_h, cfg.input_w);
        }
#else
        std::cout << "OpenCV not available, using test input..." << std::endl;
        input = create_test_input(device, cfg.batch_size, 3, cfg.input_h, cfg.input_w);
#endif
        std::cout << "Input shape: " << input.shape().to_string() << std::endl;
        std::cout << "Batch size: " << cfg.batch_size << std::endl;
        
        // 设置输入
        std::cout << "Setting inputs..." << std::endl;
        graph.set_inputs("pnnx_input_0", {input});
        
        // 执行推理
        std::cout << "Running inference..." << std::endl;
        graph.forward(cfg.debug);  // 根据配置决定是否输出调试信息
        
        // 获取输出
        std::cout << "Getting outputs..." << std::endl;
        auto outputs = graph.get_outputs("pnnx_output_0");
        
        if (outputs.empty())
        {
            std::cout << "Warning: No outputs received!" << std::endl;
            return;
        }
        
        logi("Inference successful! Output count: {}", outputs.size());
        std::cout << "Inference successful!" << std::endl;
        std::cout << "Output shape: " << outputs[0].shape().to_string() << std::endl;
        
        // 输出推理结果的统计信息，用于调试
        auto output = outputs[0];
        auto output_data = output.to_vector<float>();
        auto output_shape = output.shape();
        
        // 保存 YoloDetect 层的输出到文件，用于与 KuiperInferGitee 对比
        {
            std::ofstream out_file("tmp/output/origindl_yolo_output.txt");
            if (out_file.is_open()) {
                out_file << std::fixed << std::setprecision(8);
                out_file << "Output shape: " << output_shape.to_string() << "\n";
                out_file << "Total elements: " << output_data.size() << "\n\n";
                
                // 保存所有数据
                for (size_t i = 0; i < output_data.size(); ++i) {
                    out_file << output_data[i];
                    if ((i + 1) % 10 == 0) out_file << "\n";
                    else out_file << " ";
                }
                out_file << "\n";
                out_file.close();
                std::cout << "YoloDetect output saved to: tmp/output/origindl_yolo_output.txt" << std::endl;
            }
        }
        
        std::cout << "\n=== Inference Output Statistics ===" << std::endl;
        std::cout << "Output total elements: " << output_data.size() << std::endl;
        
        if (!output_data.empty()) {
            float min_val = *std::min_element(output_data.begin(), output_data.end());
            float max_val = *std::max_element(output_data.begin(), output_data.end());
            float sum_val = std::accumulate(output_data.begin(), output_data.end(), 0.0f);
            float mean_val = sum_val / output_data.size();
            
            std::cout << "Min value: " << min_val << std::endl;
            std::cout << "Max value: " << max_val << std::endl;
            std::cout << "Mean value: " << mean_val << std::endl;
            std::cout << "Sum value: " << sum_val << std::endl;
            
            // 输出前20个值（用于对比）
            std::cout << "\nFirst 20 output values:" << std::endl;
            size_t print_count = std::min(size_t(20), output_data.size());
            for (size_t i = 0; i < print_count; ++i) {
                std::cout << "  [" << i << "] = " << std::fixed << std::setprecision(6) 
                          << output_data[i];
                if ((i + 1) % 5 == 0) std::cout << std::endl;
                else std::cout << ", ";
            }
            if (print_count % 5 != 0) std::cout << std::endl;
            
            // 输出每个 batch 的第一个检测框的数据（前85个值）
            if (output_shape.size() == 3) {
                size_t batch_size = output_shape[0];
                size_t elements = output_shape[1];
                size_t num_info = output_shape[2];
                std::cout << "\nFirst detection box data (batch 0, element 0, all " 
                          << num_info << " values):" << std::endl;
                for (size_t j = 0; j < num_info; ++j) {
                    size_t idx = j;
                    std::cout << "  [" << j << "] = " << std::fixed << std::setprecision(6) 
                              << output_data[idx];
                    if ((j + 1) % 5 == 0) std::cout << std::endl;
                    else std::cout << ", ";
                }
                if (num_info % 5 != 0) std::cout << std::endl;
            }
        }
        std::cout << "===================================\n" << std::endl;
        
#ifdef OPENCV_FOUND
        // 后处理和绘制检测框
        if (!cfg.image_path.empty()) {
            cv::Mat image = cv::imread(cfg.image_path);
            if (!image.empty()) {
                const int32_t origin_input_h = image.rows;
                const int32_t origin_input_w = image.cols;
                const int32_t input_h = cfg.input_h;
                const int32_t input_w = cfg.input_w;
                
                auto output = outputs[0];
                auto output_shape = output.shape();
                
                // 处理 batch 输出：如果 batch_size > 1，只处理第一个 batch 的结果
                // 输出形状可能是 (batch_size, 25200, 85) 或 (25200, 85)
                size_t batch_idx = 0;  // 只处理第一个 batch
                
                const size_t elements = (output_shape.size() == 3) ? output_shape[1] : output_shape[0];  // 25200
                const size_t num_info = (output_shape.size() == 3) ? output_shape[2] : output_shape[1];  // 85
                
                auto output_data = output.to_vector<float>();
                
                // 计算当前 batch 的起始索引
                size_t batch_offset = (output_shape.size() == 3) ? batch_idx * elements * num_info : 0;
                
                std::vector<cv::Rect> boxes;
                std::vector<float> confs;
                std::vector<int> class_ids;
                
                const float confidence_thresh = cfg.confidence_thresh;
                const float iou_thresh = cfg.iou_thresh;
                
                // 解析检测结果（只处理第一个 batch）
                // 注意：YOLOv5 输出格式为 [x_center, y_center, width, height, objectness, class1, class2, ...]
                for (size_t e = 0; e < elements; ++e) {
                    size_t base_idx = batch_offset + e * num_info;
                    float cls_conf = output_data[base_idx + 4];  // objectness
                    
                    // 使用 objectness 进行过滤（与 KuiperInferGitee 保持一致）
                    if (cls_conf >= confidence_thresh) {
                        // 读取坐标（已经是模型输入尺寸下的坐标）
                        // 注意：与 KuiperInferGitee 保持一致，直接使用 int 类型转换
                        int center_x = (int)(output_data[base_idx + 0]);
                        int center_y = (int)(output_data[base_idx + 1]);
                        int width = (int)(output_data[base_idx + 2]);
                        int height = (int)(output_data[base_idx + 3]);
                        int left = center_x - width / 2;
                        int top = center_y - height / 2;
                        
                        // 找到最佳类别
                        int best_class_id = -1;
                        float best_conf = -1.f;
                        for (size_t j = 5; j < num_info; ++j) {
                            if (output_data[base_idx + j] > best_conf) {
                                best_conf = output_data[base_idx + j];
                                best_class_id = int(j - 5);
                            }
                        }
                        
                        // 计算最终置信度（类别分数 * objectness）
                        float final_conf = best_conf * cls_conf;
                        
                        // 使用最终置信度进行二次过滤（与 KuiperInferGitee 保持一致）
                        // KuiperInferGitee 在 NMSBoxes 中使用 conf_thresh 作为 score_threshold
                        if (final_conf < confidence_thresh) {
                            continue;
                        }
                        
                        // 过滤掉太小的检测框（可能是误检）
                        // 在输入尺寸（320x320）下，最小检测框应该至少 5x5 像素
                        const int min_box_size = 5;  // 最小检测框尺寸
                        if (width < min_box_size || height < min_box_size) {
                            continue;
                        }
                        
                        // 同时过滤掉面积太小的检测框（面积 < 25 像素）
                        if (width * height < 25) {
                            continue;
                        }
                        
                        // 裁剪检测框坐标到有效范围（与 KuiperInferGitee 保持一致）
                        // 注意：KuiperInferGitee 不进行有效性检查，直接添加检测框
                        // 但我们需要确保坐标在有效范围内
                        left = std::max(0, std::min(left, input_w - 1));
                        top = std::max(0, std::min(top, input_h - 1));
                        width = std::max(1, std::min(width, input_w - left));
                        height = std::max(1, std::min(height, input_h - top));
                        
                        // 验证检测框坐标是否有效
                        if (width > 0 && height > 0) {
                            boxes.emplace_back(left, top, width, height);
                            confs.emplace_back(final_conf);
                            class_ids.emplace_back(best_class_id);
                        }
                    }
                }
                
                logi("Before NMS: {} boxes (confidence >= {})", boxes.size(), confidence_thresh);
                
                // 调试：打印前几个检测框的信息
                if (boxes.size() > 0 && boxes.size() <= 5) {
                    std::cout << "\n=== DEBUG: First few boxes before NMS ===" << std::endl;
                    for (size_t i = 0; i < boxes.size(); ++i) {
                        std::cout << "Box " << i << ": left=" << boxes[i].x << ", top=" << boxes[i].y 
                                  << ", width=" << boxes[i].width << ", height=" << boxes[i].height
                                  << ", conf=" << confs[i] << ", class=" << class_ids[i] << std::endl;
                    }
                }

                // NMS: score_threshold 用于过滤低置信度的检测框
                // 与 KuiperInferGitee 保持一致，使用 confidence_thresh 作为 score_threshold
                // 这样可以在 NMS 阶段进一步过滤低置信度的检测框
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confs, confidence_thresh, iou_thresh, indices);
                
                logi("After NMS: {} boxes", indices.size());
                
                // 绘制检测框
                std::vector<Detection> detections;
                std::cout << "\n=== DEBUG: Detection Boxes After NMS ===" << std::endl;
                std::cout << "Input image size: " << input_w << "x" << input_h << std::endl;
                std::cout << "Original image size: " << origin_input_w << "x" << origin_input_h << std::endl;
                for (int idx : indices) {
                    Detection det;
                    det.box = cv::Rect(boxes[idx]);
                    std::cout << "Box " << idx << " (before ScaleCoords): left=" << det.box.x 
                              << ", top=" << det.box.y << ", width=" << det.box.width 
                              << ", height=" << det.box.height << ", conf=" << confs[idx] 
                              << ", class=" << class_ids[idx] << std::endl;
                    ScaleCoords(cv::Size{input_w, input_h}, det.box, 
                               cv::Size{origin_input_w, origin_input_h});
                    std::cout << "Box " << idx << " (after ScaleCoords): left=" << det.box.x 
                              << ", top=" << det.box.y << ", width=" << det.box.width 
                              << ", height=" << det.box.height << std::endl;
                    det.conf = confs[idx];
                    det.class_id = class_ids[idx];
                    detections.emplace_back(det);
                }
                
                int font_face = cv::FONT_HERSHEY_COMPLEX;
                double font_scale = 1.0;
                int thickness = 2;
                
                for (const auto& detection : detections) {
                    cv::rectangle(image, detection.box, cv::Scalar(0, 255, 0), thickness);
                    std::string label = "Class " + std::to_string(detection.class_id) + 
                                       " (" + std::to_string(detection.conf).substr(0, 4) + ")";
                    cv::putText(image, label,
                               cv::Point(detection.box.x, detection.box.y - 5), 
                               font_face, font_scale,
                               cv::Scalar(0, 255, 0), thickness);
                }
                
                // 输出路径：如果未指定，保存到 tmp 目录
                std::string output_path = cfg.output_path.empty() ? "tmp/output_detection.jpg" : cfg.output_path;
                // 确保 tmp 目录存在
                if (output_path.find("tmp/") == 0 || output_path.find("./tmp/") == 0) {
                    std::string cmd = "mkdir -p tmp";
                    int ret = std::system(cmd.c_str());
                    (void)ret;  // 忽略返回值
                }
                cv::imwrite(output_path, image);
                std::cout << "Detection results saved to: " << output_path << std::endl;
                std::cout << "Found " << detections.size() << " objects" << std::endl;
            }
        }
#endif
        
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
    UserCfg cfg = parse_args(argc, argv);
    
    if (cfg.show_help) {
        print_help(argv[0]);
        return 0;
    }
    
    if (cfg.param_path.empty() || cfg.bin_path.empty()) {
        std::cerr << "Error: param_path and bin_path are required!" << std::endl;
        print_help(argv[0]);
        return 1;
    }
    
    // 从 param 文件中自动解析输入形状（如果未通过命令行指定）
    uint32_t parsed_batch_size;
    int32_t parsed_input_h, parsed_input_w;
    if (GetInputShapeFromParamFile(cfg.param_path, parsed_batch_size, parsed_input_h, parsed_input_w)) {
        // 如果命令行没有指定 batch_size，使用解析出的值
        if (cfg.batch_size == 1) {
            cfg.batch_size = static_cast<int>(parsed_batch_size);
        }
        // 如果使用默认值（640x640），使用解析出的值
        if (cfg.input_h == 640 && cfg.input_w == 640) {
            cfg.input_h = parsed_input_h;
            cfg.input_w = parsed_input_w;
        }
        std::cout << "Using batch_size=" << cfg.batch_size << ", input_h=" << cfg.input_h 
                  << ", input_w=" << cfg.input_w << std::endl;
    }
    
    try
    {
        YoloDemo(cfg);
        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
