// YOLOv5 推理示例（使用 PNNX 模型）
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
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
    float confidence_thresh = 0.25f; // 置信度阈值
    float iou_thresh = 0.45f;        // IOU 阈值
    int gpu_device = 0;              // GPU 设备 ID（默认使用 gpu0）
    bool debug = false;              // 是否输出调试信息
    bool show_help = false;          // 是否显示帮助信息
};

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
        {"debug",      no_argument,       0, 'd'},
        {"help",       no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    
    int option_index = 0;
    int c;
    
    while ((c = getopt_long(argc, argv, "p:b:i:o:c:u:g:dh", long_options, &option_index)) != -1) {
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
              << "  -d, --debug           Enable debug logging\n"
              << "  -h, --help            Show this help message\n"
              << "\n"
              << "Examples:\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i image.jpg\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i image.jpg -c 0.5 -u 0.5\n"
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
    // Tensor 格式：CHW (Channel, Height, Width)
    // OpenCV Mat 格式：HWC (Height, Width, Channel)
    for (int c = 0; c < input_c; ++c) {
        const cv::Mat& channel = split_images[c];
        // 转置：从 (H, W) 到 (W, H)，然后按行存储
        cv::Mat transposed = channel.t();
        memcpy(input_data.data() + c * input_h * input_w, 
               transposed.data, 
               sizeof(float) * input_h * input_w);
    }
    
    Shape input_shape{1, static_cast<size_t>(input_c), 
                     static_cast<size_t>(input_h), static_cast<size_t>(input_w)};
    
    return Tensor(input_data, input_shape, dtype(DataType::kFloat32).device(device));
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
    
    // 确定设备
    Device device(DeviceType::kCPU);
#ifdef WITH_CUDA
    if (cuda::is_available())
    {
        device = Device(DeviceType::kCUDA, cfg.gpu_device);
        logi("CUDA is available. Using GPU{} for inference.", cfg.gpu_device);
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
        
        // 创建输入
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
            input = preprocess_image(image, device);
        } else {
            std::cout << "No image path provided, using test input..." << std::endl;
            input = create_test_input(device, 1, 3, 640, 640);
        }
#else
        std::cout << "OpenCV not available, using test input..." << std::endl;
        input = create_test_input(device, 1, 3, 640, 640);
#endif
        std::cout << "Input shape: " << input.shape().to_string() << std::endl;
        
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
        
#ifdef OPENCV_FOUND
        // 后处理和绘制检测框
        if (!cfg.image_path.empty()) {
            cv::Mat image = cv::imread(cfg.image_path);
            if (!image.empty()) {
                const int32_t origin_input_h = image.rows;
                const int32_t origin_input_w = image.cols;
                const int32_t input_h = 640;
                const int32_t input_w = 640;
                
                auto output = outputs[0];
                auto output_shape = output.shape();
                const size_t elements = output_shape[1];  // 25200
                const size_t num_info = output_shape[2];  // 85
                
                auto output_data = output.to_vector<float>();
                
                std::vector<cv::Rect> boxes;
                std::vector<float> confs;
                std::vector<int> class_ids;
                
                const float confidence_thresh = cfg.confidence_thresh;
                const float iou_thresh = cfg.iou_thresh;
                
                // 解析检测结果
                for (size_t e = 0; e < elements; ++e) {
                    size_t base_idx = e * num_info;
                    float cls_conf = output_data[base_idx + 4];  // objectness
                    
                    // 先用 objectness 进行初步过滤（提高效率）
                    if (cls_conf >= confidence_thresh) {
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
                        
                        // 只保留最终置信度大于阈值的检测框
                        if (final_conf >= confidence_thresh) {
                            boxes.emplace_back(left, top, width, height);
                            confs.emplace_back(final_conf);
                            class_ids.emplace_back(best_class_id);
                        }
                    }
                }
                
                logi("Before NMS: {} boxes (confidence >= {})", boxes.size(), confidence_thresh);

                // NMS: score_threshold 用于过滤低置信度的检测框
                // 由于我们已经在前面用 confidence_thresh 过滤过了，这里可以传入 0.0
                // 但为了安全起见，仍然传入 confidence_thresh 作为双重过滤
                std::vector<int> indices;
                cv::dnn::NMSBoxes(boxes, confs, confidence_thresh, iou_thresh, indices);
                
                logi("After NMS: {} boxes", indices.size());
                
                // 绘制检测框
                std::vector<Detection> detections;
                for (int idx : indices) {
                    Detection det;
                    det.box = cv::Rect(boxes[idx]);
                    ScaleCoords(cv::Size{input_w, input_h}, det.box, 
                               cv::Size{origin_input_w, origin_input_h});
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
                
                std::string output_path = cfg.output_path.empty() ? "output_detection.jpg" : cfg.output_path;
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
