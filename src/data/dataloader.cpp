#include "origin/data/dataloader.h"
#include <algorithm>
#include <random>
#include "origin/core/tensor.h"
#include "origin/utils/exception.h"

namespace origin
{

DataLoader::DataLoader(Dataset &dataset, size_t batch_size, bool shuffle)
    : dataset_(&dataset), batch_size_(batch_size), shuffle_(shuffle), current_index_(0)
{
    reset_indices();
}

void DataLoader::reset_indices()
{
    indices_.clear();
    indices_.reserve(dataset_->size());
    for (size_t i = 0; i < dataset_->size(); ++i)
    {
        indices_.push_back(i);
    }

    if (shuffle_)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_.begin(), indices_.end(), g);
    }

    current_index_ = 0;
}

std::pair<Tensor, Tensor> DataLoader::next()
{
    if (!has_next())
    {
        // 对于空数据集，抛出异常或返回特殊值
        // 实际使用中，应该在调用 next() 之前检查 has_next()
        THROW_RUNTIME_ERROR(
            "DataLoader::next() called when no more data available. Use has_next() to check before calling next().");
    }

    // 确定实际批大小（可能是最后一个不完整的批次）
    size_t actual_batch_size = std::min(batch_size_, indices_.size() - current_index_);

    // 收集批次数据
    std::vector<std::vector<float>> batch_images;
    std::vector<float> batch_labels;
    batch_images.reserve(actual_batch_size);
    batch_labels.reserve(actual_batch_size);

    // 获取第一个样本以确定输入维度
    size_t first_idx                = indices_[current_index_];
    auto [first_image, first_label] = dataset_->get_item(first_idx);
    auto first_image_shape          = first_image.shape();
    size_t input_size               = first_image_shape.elements();  // 输入的总元素数（例如 784 或 1）

    for (size_t i = 0; i < actual_batch_size; ++i)
    {
        size_t idx          = indices_[current_index_ + i];
        auto [image, label] = dataset_->get_item(idx);

        // 验证图像形状一致
        if (image.shape().elements() != input_size)
        {
            THROW_RUNTIME_ERROR("Inconsistent input shapes in dataset: expected {} elements, got {}", input_size,
                                image.shape().elements());
        }

        // 将图像添加到批次
        auto image_data = image.to_vector<float>();
        batch_images.push_back(image_data);

        // 将标签添加到批次
        batch_labels.push_back(label.item<float>());
    }

    // 创建批次张量
    // inputs: (batch_size, input_size)
    std::vector<float> inputs_flat;
    inputs_flat.reserve(actual_batch_size * input_size);
    for (const auto &img : batch_images)
    {
        inputs_flat.insert(inputs_flat.end(), img.begin(), img.end());
    }
    Shape input_shape{actual_batch_size, input_size};
    auto inputs = Tensor(inputs_flat, input_shape, dtype(DataType::kFloat32));

    // targets: (batch_size,)
    auto targets = Tensor(batch_labels, Shape{actual_batch_size}, dtype(DataType::kFloat32));

    // 更新当前索引
    current_index_ += actual_batch_size;

    return std::make_pair(inputs, targets);
}

bool DataLoader::has_next() const
{
    return current_index_ < indices_.size();
}

void DataLoader::reset()
{
    reset_indices();
}

}  // namespace origin
