/* 
YOLOv5 推理示例（使用 PNNX 模型）
用法示例：
./build/bin/example/example_yolov5 -i data/imgs/ -o data/outputs/ -p model/yolo/yolov5n_small.pnnx.param -b model/yolo/yolov5n_small.pnnx.bin 
*/
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
#include <filesystem>
#include "origin.h"
#include "origin/pnnx/pnnx_graph.h"
#include "origin/utils/log.h"
#include "origin/core/config.h"
#include "class_labels.h"
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
    std::string image_dir;           // 输入图像目录路径
    std::string output_dir;          // 输出图像目录路径
    float confidence_thresh = 0.25f; // 置信度阈值
    float iou_thresh = 0.25f;        // IOU 阈值
    int gpu_device = 0;              // GPU 设备 ID（默认使用 gpu0）
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
bool get_input_shape_from_param_file(const std::string& param_path, 
                                uint32_t& batch_size, 
                                int32_t& input_h, 
                                int32_t& input_w) {
    std::ifstream file(param_path);
    if (!file.is_open()) {
        logw("Failed to open param file: {}, using default values", param_path);
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
                        logi("Parsed input shape from param file: batch_size={}, input_h={}, input_w={}", 
                             batch_size, input_h, input_w);
                        return true;
                    } catch (const std::exception& e) {
                        logw("Failed to parse shape dimensions, error: {}, using default values", e.what());
                    }
                }
            }
        }
    }
    
    logw("Failed to parse input shape from param file: {}, using default values", param_path);
    batch_size = 1;
    input_h = 640;
    input_w = 640;
    return false;
}

/**
 * @brief 从目录中获取所有图像文件路径
 * @param dir_path 目录路径
 * @return 图像文件路径列表（已排序）
 */
std::vector<std::string> get_image_files_from_directory(const std::string& dir_path) {
    std::vector<std::string> image_files;
    
    if (!std::filesystem::exists(dir_path) || !std::filesystem::is_directory(dir_path)) {
        logw("Directory does not exist or is not a directory: {}", dir_path);
        return image_files;
    }
    
    // 支持的图像扩展名
    std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"};
    
    for (const auto& entry : std::filesystem::directory_iterator(dir_path)) {
        if (entry.is_regular_file()) {
            std::string file_path = entry.path().string();
            std::string extension = entry.path().extension().string();
            
            // 检查是否是图像文件
            if (std::find(image_extensions.begin(), image_extensions.end(), extension) != image_extensions.end()) {
                image_files.push_back(file_path);
            }
        }
    }
    
    // 按文件名排序，保证处理顺序一致
    std::sort(image_files.begin(), image_files.end());
    
    return image_files;
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
                cfg.image_dir = optarg;
                break;
            case 'o':
                cfg.output_dir = optarg;
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
    if (cfg.image_dir.empty() && optind < argc) {
        cfg.image_dir = argv[optind++];
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
              << "  -i, --image DIR       Input image directory path (default: use test input)\n"
              << "  -o, --output DIR      Output image directory path (default: ./tmp)\n"
              << "  -c, --confidence FLOAT Confidence threshold (default: 0.25)\n"
              << "  -u, --iou FLOAT       IOU threshold for NMS (default: 0.45)\n"
              << "  -g, --gpu INT         GPU device ID (default: 0)\n"
              << "  -d, --debug           Enable debug logging\n"
              << "  -h, --help            Show this help message\n"
              << "\n"
              << "Examples:\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i ./images -o ./output\n"
              << "  " << program_name << " -p model.pnnx.param -b model.pnnx.bin -i ./images -o ./output -c 0.5 -u 0.5\n"
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
float letterbox(const cv::Mat& image, cv::Mat& out_image, 
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
void scale_coords(const cv::Size& img_shape, cv::Rect& coords, const cv::Size& img_origin_shape) {
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
 * @brief 图像预处理：letterbox + 归一化 + BGR2RGB + 转换为 Tensor
 */
Tensor preprocess_image(const cv::Mat& image, const Device& device, int input_h = 640, int input_w = 640) {
    const int input_c = 3;
    
    // letterbox
    cv::Mat out_image;
    letterbox(image, out_image, {input_h, input_w}, 32, {114, 114, 114}, true);
    
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
    
    // if (cfg.debug) {
    //     auto data_vec = result.to_vector<float>();
    //     float min_val = *std::min_element(data_vec.begin(), data_vec.end());
    //     float max_val = *std::max_element(data_vec.begin(), data_vec.end());
    //     float mean_val = std::accumulate(data_vec.begin(), data_vec.end(), 0.0f) / data_vec.size();
    //     logd("Preprocessed Input: shape={}, min={}, max={}, mean={}", 
    //          result.shape().to_string(), min_val, max_val, mean_val);
    // }
    
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

#ifdef OPENCV_FOUND
/**
 * @brief 处理单张图像的检测结果并保存
 * @param output_data 模型输出数据
 * @param output_shape 输出形状
 * @param batch_idx batch索引
 * @param image 原始图像
 * @param image_path 原始图像路径
 * @param output_path 输出图像路径
 * @param cfg 配置参数
 * @param class_names 类别名称列表（如果为空则使用数字ID）
 */
void process_and_save_detection(const std::vector<float>& output_data,
                                 const Shape& output_shape,
                                 size_t batch_idx,
                                 const cv::Mat& image,
                                 const std::string& image_path,
                                 const std::string& output_path,
                                 const UserCfg& cfg,
                                 const std::vector<std::string>& class_names) {
    const int32_t origin_input_h = image.rows;
    const int32_t origin_input_w = image.cols;
    const int32_t input_h = cfg.input_h;
    const int32_t input_w = cfg.input_w;
    
    // 输出形状可能是 (batch_size, 25200, 85) 或 (25200, 85)
    const size_t elements = (output_shape.size() == 3) ? output_shape[1] : output_shape[0];  // 25200
    const size_t num_info = (output_shape.size() == 3) ? output_shape[2] : output_shape[1];  // 85
    
    // 计算当前 batch 的起始索引
    size_t batch_offset = (output_shape.size() == 3) ? batch_idx * elements * num_info : 0;
    
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> class_ids;
    
    const float confidence_thresh = cfg.confidence_thresh;
    const float iou_thresh = cfg.iou_thresh;
    
    // 解析检测结果
    // 注意：YOLOv5 输出格式为 [x_center, y_center, width, height, objectness, class1, class2, ...]
    for (size_t e = 0; e < elements; ++e) {
        size_t base_idx = batch_offset + e * num_info;
        float cls_conf = output_data[base_idx + 4];  // objectness
        
        // 使用 objectness 进行过滤（与 KuiperInferGitee 保持一致）
        if (cls_conf >= confidence_thresh) {
            // 读取坐标（已经是模型输入尺寸下的坐标）
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
            
            // 使用最终置信度进行二次过滤
            if (final_conf < confidence_thresh) {
                continue;
            }
            
            // 过滤掉太小的检测框
            const int kMinBoxSize = 5;
            if (width < kMinBoxSize || height < kMinBoxSize) {
                continue;
            }
            
            // 同时过滤掉面积太小的检测框
            if (width * height < 25) {
                continue;
            }
            
            // 裁剪检测框坐标到有效范围
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
    
    logi("Image {}: Before NMS: {} boxes (confidence >= {})", 
         std::filesystem::path(image_path).filename().string(), boxes.size(), confidence_thresh);
    
    // NMS
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confs, confidence_thresh, iou_thresh, indices);
    
    logi("Image {}: After NMS: {} boxes", 
         std::filesystem::path(image_path).filename().string(), indices.size());
    
    // 绘制检测框
    cv::Mat result_image = image.clone();
    std::vector<Detection> detections;
    for (int idx : indices) {
        Detection det;
        det.box = cv::Rect(boxes[idx]);
        scale_coords(cv::Size{input_w, input_h}, det.box, 
                   cv::Size{origin_input_w, origin_input_h});
        det.conf = confs[idx];
        det.class_id = class_ids[idx];
        detections.emplace_back(det);
    }
    
    // 定义10种颜色（BGR格式）：用于不同类别的检测框
    // 颜色ID: 0-9，类别ID % 10 = 颜色ID
    const std::vector<cv::Scalar> colors = {
        cv::Scalar(0, 255, 0),      // 0: 绿色
        cv::Scalar(255, 0, 0),      // 1: 蓝色
        cv::Scalar(0, 0, 255),      // 2: 红色
        cv::Scalar(255, 255, 0),    // 3: 青色
        cv::Scalar(255, 0, 255),    // 4: 洋红色
        cv::Scalar(0, 255, 255),    // 5: 黄色
        cv::Scalar(128, 0, 128),    // 6: 紫色
        cv::Scalar(255, 165, 0),    // 7: 橙色
        cv::Scalar(0, 128, 255),    // 8: 橙红色
        cv::Scalar(128, 255, 0)     // 9: 黄绿色
    };
    
    int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double default_font_scale = 0.6;
    int thickness = 2;
    int baseline = 0;
    const int text_padding = 3;
    
    for (const auto& detection : detections) {
        // 根据类别ID选择颜色：类别ID % 10 = 颜色ID
        int color_id = detection.class_id % 10;
        cv::Scalar box_color = colors[color_id];
        
        cv::rectangle(result_image, detection.box, box_color, thickness);
        
        // 构建标签：使用类别名称（如果可用）或数字ID
        std::string label;
        if (!class_names.empty() && detection.class_id >= 0 && 
            static_cast<size_t>(detection.class_id) < class_names.size()) {
            label = class_names[detection.class_id] + " " + 
                   std::to_string(detection.conf).substr(0, 4);
        } else {
            label = "Class " + std::to_string(detection.class_id) + 
                   " (" + std::to_string(detection.conf).substr(0, 4) + ")";
        }
        
        // 为当前检测框计算合适的字体大小
        double current_font_scale = default_font_scale;
        cv::Size text_size = cv::getTextSize(label, font_face, current_font_scale, thickness, &baseline);
        
        // 计算文字位置：框内左上角，留出一些边距
        int text_x = detection.box.x + text_padding;
        int text_y = detection.box.y + text_size.height + text_padding;
        
        // 如果文字会超出检测框，调整字体大小或截断文字
        int max_width = detection.box.width - 2 * text_padding;
        if (max_width > 0 && text_size.width > max_width) {
            // 文字太宽，尝试缩小字体
            while (text_size.width > max_width && current_font_scale > 0.3) {
                current_font_scale -= 0.1;
                text_size = cv::getTextSize(label, font_face, current_font_scale, thickness, &baseline);
            }
            
            // 如果还是太宽，截断文字
            if (text_size.width > max_width) {
                // 估算能显示的字符数
                double char_width = text_size.width / static_cast<double>(label.length());
                int max_chars = static_cast<int>(max_width / char_width) - 3;
                if (max_chars > 0 && max_chars < static_cast<int>(label.length())) {
                    label = label.substr(0, max_chars) + "...";
                    text_size = cv::getTextSize(label, font_face, current_font_scale, thickness, &baseline);
                }
            }
        }
        
        // 确保文字不会超出检测框的下边界
        if (text_y > detection.box.y + detection.box.height) {
            text_y = detection.box.y + detection.box.height - text_padding;
        }
        
        // 绘制文字背景框（半透明背景）
        cv::Rect text_bg_rect(
            text_x - text_padding,
            text_y - text_size.height - text_padding,
            text_size.width + 2 * text_padding,
            text_size.height + baseline + 2 * text_padding
        );
        
        // 确保背景框在检测框内
        text_bg_rect.x = std::max(text_bg_rect.x, detection.box.x);
        text_bg_rect.y = std::max(text_bg_rect.y, detection.box.y);
        text_bg_rect.width = std::min(text_bg_rect.width, 
                                      detection.box.x + detection.box.width - text_bg_rect.x);
        text_bg_rect.height = std::min(text_bg_rect.height,
                                       detection.box.y + detection.box.height - text_bg_rect.y);
        
        // 绘制半透明背景
        cv::Mat overlay = result_image.clone();
        cv::rectangle(overlay, text_bg_rect, cv::Scalar(0, 0, 0), -1);
        cv::addWeighted(overlay, 0.5, result_image, 0.5, 0, result_image);
        
        // 绘制文字（在框内左上角），使用与检测框相同的颜色
        cv::putText(result_image, label,
                   cv::Point(text_x, text_y), 
                   font_face, current_font_scale,
                   box_color, thickness);
    }
    
    // 保存结果
    cv::imwrite(output_path, result_image);
    logi("Detection results saved to: {} (Found {} objects)", output_path, detections.size());
}
#endif  // OPENCV_FOUND

/**
 * @brief YOLOv5 推理示例
 */
void yolo_demo(const UserCfg &cfg, int batch_size)
{
    logi("=== YOLOv5 Inference Demo ===");
    logi("Loading model from:");
    logi("  Param: {}", cfg.param_path);
    logi("  Bin: {}", cfg.bin_path);
    
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
    
    logi("Configuration:");
    logi("  Confidence threshold: {}", cfg.confidence_thresh);
    logi("  IOU threshold: {}", cfg.iou_thresh);
    logi("  Device: {}", device.to_string());
    if (!cfg.image_dir.empty()) {
        logi("  Input image directory: {}", cfg.image_dir);
    }
    if (!cfg.output_dir.empty()) {
        logi("  Output image directory: {}", cfg.output_dir);
    }
    
    try
    {
        // 创建 PNNX 图
        PNNXGraph graph(cfg.param_path, cfg.bin_path);
        
        // 构建计算图
        logi("Building graph...");
        graph.build();
        logi("Graph built successfully!");
        
        // 获取图像文件列表
        std::vector<std::string> image_files;
#ifdef OPENCV_FOUND
        if (!cfg.image_dir.empty()) {
            image_files = get_image_files_from_directory(cfg.image_dir);
            if (image_files.empty()) {
                loge("No image files found in directory: {}", cfg.image_dir);
                return;
            }
            logi("Found {} image files in directory", image_files.size());
        }
#endif
        
        // 创建输出目录
        std::string output_dir = cfg.output_dir.empty() ? "./tmp" : cfg.output_dir;
        std::filesystem::create_directories(output_dir);
        logi("Output directory: {}", output_dir);
        
        // 使用COCO类别名称（从class_labels.h）
        const std::vector<std::string>& class_names = COCO_CLASSES;
        
        // 如果没有图像文件，使用测试输入
        if (image_files.empty()) {
            logi("No image directory provided, using test input...");
            Tensor input = create_test_input(device, batch_size, 3, cfg.input_h, cfg.input_w);
            logi("Input shape: {}", input.shape().to_string());
            logi("Batch size: {}", batch_size);
            
            graph.set_inputs("pnnx_input_0", {input});
            logi("Running inference...");
            graph.forward(cfg.debug);
            
            auto outputs = graph.get_outputs("pnnx_output_0");
            if (outputs.empty()) {
                logw("No outputs received!");
                return;
            }
            logi("Inference successful! Output count: {}", outputs.size());
        } else {
            // 批量处理图像
#ifdef OPENCV_FOUND
            // 按 batch_size 分批处理
            for (size_t batch_start = 0; batch_start < image_files.size(); batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, image_files.size());
                size_t actual_batch_size = batch_end - batch_start;
                
                logi("Processing batch: images {} to {} ({} images)", 
                     batch_start + 1, batch_end, actual_batch_size);
                
                // 加载当前批次的图像
                std::vector<cv::Mat> batch_images;
                std::vector<std::string> batch_image_paths;
                for (size_t i = batch_start; i < batch_end; ++i) {
                    cv::Mat image = cv::imread(image_files[i]);
                    if (image.empty()) {
                        logw("Cannot load image: {}, skipping", image_files[i]);
                        continue;
                    }
                    batch_images.push_back(image);
                    batch_image_paths.push_back(image_files[i]);
                }
                
                // 如果当前批次没有有效图像，跳过
                if (batch_images.empty()) {
                    logw("No valid images in current batch, skipping");
                    continue;
                }
                
                // 如果实际图像数量小于 batch_size，用最后一张图像填充
                size_t current_batch_size = batch_images.size();
                if (current_batch_size < static_cast<size_t>(batch_size)) {
                    logi("Padding batch from {} to {} images (using last image)", 
                         current_batch_size, batch_size);
                    cv::Mat last_image = batch_images.back();
                    std::string last_path = batch_image_paths.back();
                    for (size_t i = current_batch_size; i < static_cast<size_t>(batch_size); ++i) {
                        batch_images.push_back(last_image.clone());
                        batch_image_paths.push_back(last_path);  // 用于填充，但不会保存输出
                    }
                }
                
                // 预处理所有图像并合并为 batch
                std::vector<Tensor> preprocessed_images;
                for (const auto& image : batch_images) {
                    Tensor preprocessed = preprocess_image(image, device, cfg.input_h, cfg.input_w);
                    preprocessed_images.push_back(preprocessed);
                }
                
                // 合并为 batch tensor
                if (preprocessed_images.empty()) {
                    continue;
                }
                
                Shape single_shape = preprocessed_images[0].shape();
                size_t single_size = single_shape.elements();
                std::vector<float> batch_data(static_cast<size_t>(batch_size) * single_size);
                
                for (size_t i = 0; i < static_cast<size_t>(batch_size); ++i) {
                    auto single_data = preprocessed_images[i].to_vector<float>();
                    std::copy(single_data.begin(), single_data.end(), 
                             batch_data.begin() + i * single_size);
                }
                
                std::vector<size_t> batch_dims = single_shape.dims();
                batch_dims[0] = static_cast<size_t>(batch_size);
                Tensor input = Tensor(batch_data, Shape(batch_dims), 
                                      dtype(DataType::kFloat32).device(device));
                
                logi("Input shape: {}", input.shape().to_string());
                
                // 设置输入并执行推理
                graph.set_inputs("pnnx_input_0", {input});
                logi("Running inference...");
                graph.forward(cfg.debug);
                
                // 获取输出
                auto outputs = graph.get_outputs("pnnx_output_0");
                if (outputs.empty()) {
                    logw("No outputs received for batch!");
                    continue;
                }
                
                logi("Inference successful! Output count: {}", outputs.size());
                
                // 处理每个 batch 中的每张图像
                auto output = outputs[0];
                auto output_shape = output.shape();
                auto output_data = output.to_vector<float>();
                
                // 只处理实际图像（不包括填充的图像）
                for (size_t i = 0; i < current_batch_size; ++i) {
                    size_t image_idx = batch_start + i;
                    std::string image_path = image_files[image_idx];
                    cv::Mat image = batch_images[i];
                    
                    // 生成输出文件名：output_ + 原始文件名
                    std::filesystem::path input_path(image_path);
                    std::string output_filename = "output_" + input_path.filename().string();
                    std::string output_path = (std::filesystem::path(output_dir) / output_filename).string();
                    
                    // 处理并保存检测结果
                    process_and_save_detection(output_data, output_shape, i, 
                                              image, image_path, output_path, cfg, class_names);
                }
            }
            
            logi("Processed {} images in total", image_files.size());
#endif  // OPENCV_FOUND
        }
        
        logi("=== YOLOv5 Inference Complete ===");
    }
    catch (const std::exception &e)
    {
        loge("Error: {}", e.what());
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
        loge("param_path and bin_path are required!");
        print_help(argv[0]);
        return 1;
    }
    
    // 从 param 文件中自动解析输入形状
    uint32_t parsed_batch_size = 1;
    int32_t parsed_input_h = 640, parsed_input_w = 640;
    if (get_input_shape_from_param_file(cfg.param_path, parsed_batch_size, parsed_input_h, parsed_input_w)) {
        // 如果使用默认值（640x640），使用解析出的值
        if (cfg.input_h == 640 && cfg.input_w == 640) {
            cfg.input_h = parsed_input_h;
            cfg.input_w = parsed_input_w;
        }
        logi("Using batch_size={}, input_h={}, input_w={}", 
             parsed_batch_size, cfg.input_h, cfg.input_w);
    } else {
        logi("Using default batch_size=1, input_h={}, input_w={}", 
             cfg.input_h, cfg.input_w);
    }
    
    try
    {
        yolo_demo(cfg, static_cast<int>(parsed_batch_size));
        return 0;
    }
    catch (const std::exception &e)
    {
        loge("Fatal error: {}", e.what());
        return 1;
    }
}
