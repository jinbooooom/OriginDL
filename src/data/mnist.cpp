#include "origin/data/mnist.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{

MNIST::MNIST(const std::string &root, bool train) : train_(train), root_(root)
{
    // 创建数据目录
    std::filesystem::create_directories(root_);

    // 确定文件名
    std::string images_file, labels_file;
    if (train_)
    {
        images_file = root_ + "/train-images-idx3-ubyte";
        labels_file = root_ + "/train-labels-idx1-ubyte";
    }
    else
    {
        images_file = root_ + "/t10k-images-idx3-ubyte";
        labels_file = root_ + "/t10k-labels-idx1-ubyte";
    }

    // Dataset 仅负责加载，用户自行调用下载脚本
    // if (!std::filesystem::exists(images_file) || !std::filesystem::exists(labels_file))
    // {
    //     std::cout << "MNIST data files not found. Running download script..." << std::endl;
    //
    //     std::filesystem::path script_path = std::filesystem::current_path() / "scripts" / "download_mnist.sh";
    //     if (unlikely(!std::filesystem::exists(script_path)))
    //     {
    //         THROW_RUNTIME_ERROR(
    //             "MNIST data files not found and download script not found at: {}\n"
    //             "Please ensure the script exists or download the data manually.",
    //             script_path.string());
    //     }
    //     std::string cmd = "bash " + script_path.string();
    //     int ret         = std::system(cmd.c_str());
    //     if (unlikely(ret != 0))
    //     {
    //         THROW_RUNTIME_ERROR(
    //             "Failed to download MNIST dataset. Download script returned error code: {}\n"
    //             "Please run the script manually: bash scripts/download_mnist.sh",
    //             ret);
    //     }
    //     if (unlikely(!std::filesystem::exists(images_file) || !std::filesystem::exists(labels_file)))
    //     {
    //         THROW_RUNTIME_ERROR(
    //             "MNIST data files still not found after running download script.\n"
    //             "Expected files: {} and {}",
    //             images_file, labels_file);
    //     }
    //     std::cout << "MNIST dataset downloaded successfully." << std::endl;
    // }

    // 加载数据
    if (unlikely(!load_images(images_file)))
    {
        THROW_RUNTIME_ERROR("Failed to load MNIST images from: {}", images_file);
    }
    if (unlikely(!load_labels(labels_file)))
    {
        THROW_RUNTIME_ERROR("Failed to load MNIST labels from: {}", labels_file);
    }
}

uint32_t MNIST::read_uint32(std::ifstream &file)
{
    uint32_t value = 0;
    for (int i = 0; i < 4; ++i)
    {
        unsigned char byte;
        file.read(reinterpret_cast<char *>(&byte), 1);
        value = (value << 8) | byte;
    }
    return value;
}

bool MNIST::load_images(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    // 读取魔数（应该为 2051）
    uint32_t magic = read_uint32(file);
    if (magic != 2051)
    {
        return false;
    }

    // 读取图像数量
    uint32_t num_images = read_uint32(file);
    // 读取图像尺寸（应该是 28x28）
    uint32_t rows = read_uint32(file);
    uint32_t cols = read_uint32(file);

    if (rows != 28 || cols != 28)
    {
        return false;
    }

    // 读取图像数据
    images_.clear();
    images_.reserve(num_images);

    for (uint32_t i = 0; i < num_images; ++i)
    {
        std::vector<float> image(rows * cols);
        for (size_t j = 0; j < rows * cols; ++j)
        {
            unsigned char pixel;
            file.read(reinterpret_cast<char *>(&pixel), 1);
            // 归一化到 [0, 1]
            image[j] = static_cast<float>(pixel) / 255.0f;
        }
        images_.push_back(std::move(image));
    }

    return true;
}

bool MNIST::load_labels(const std::string &filepath)
{
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    // 读取魔数（应该为 2049）
    uint32_t magic = read_uint32(file);
    if (magic != 2049)
    {
        return false;
    }

    // 读取标签数量
    uint32_t num_labels = read_uint32(file);

    // 读取标签数据
    labels_.clear();
    labels_.reserve(num_labels);

    for (uint32_t i = 0; i < num_labels; ++i)
    {
        unsigned char label;
        file.read(reinterpret_cast<char *>(&label), 1);
        labels_.push_back(static_cast<int32_t>(label));
    }

    return true;
}

std::pair<Tensor, Tensor> MNIST::get_item(size_t index)
{
    if (unlikely(index >= size()))
    {
        THROW_INVALID_ARG("Index {} out of range for MNIST dataset with {} samples", index, size());
    }

    // 创建图像张量 (784,)
    auto image = Tensor(images_[index], Shape{784}, dtype(DataType::kFloat32));

    // 创建标签张量 (标量, 使用 int64)
    // 这里使用 int64 是为了与 PyTorch 中 MNIST / CrossEntropyLoss 对 target 的约定保持一致，
    // 即分类标签使用 LongTensor（int64），后续 loss / accuracy / gather 等算子可以直接消费整型标签。
    std::vector<int64_t> label_data(1);
    label_data[0] = static_cast<int64_t>(labels_[index]);
    auto label    = Tensor(label_data, Shape{}, dtype(DataType::kInt64));

    return std::make_pair(image, label);
}

size_t MNIST::size() const
{
    return images_.size();
}

}  // namespace origin
