#include "origin/data/dataloader.h"
#include <algorithm>
#include <random>
#include "origin/core/tensor.h"
#include "origin/utils/branch_prediction.h"
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
    if (unlikely(!has_next()))
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
    batch_images.reserve(actual_batch_size);

    // 为了与 PyTorch 行为更一致，这里根据标签 dtype 决定批次标签的 dtype：
    // - 如果标签是整型（int64/int32/int8/uint8），则批次标签使用 int64（LongTensor 语义）
    // - 否则保持为 float32
    // 这样对于分类任务（如 MNIST），DataLoader 会直接返回整型标签张量，后续
    // softmax_cross_entropy / accuracy / gather 等算子可以直接消费整型 indices。
    std::vector<int64_t> batch_labels_int;
    std::vector<float> batch_labels_float;

    // 获取第一个样本以确定输入维度和标签类型
    size_t first_idx                = indices_[current_index_];
    auto [first_image, first_label] = dataset_->get_item(first_idx);
    auto first_image_shape          = first_image.shape();
    size_t input_size               = first_image_shape.elements();  // 输入的总元素数（例如 784 或 1）
    auto first_label_dtype          = first_label.dtype();

    bool use_int_labels = (first_label_dtype == DataType::kInt64 || first_label_dtype == DataType::kInt32 ||
                           first_label_dtype == DataType::kInt8 || first_label_dtype == DataType::kUInt8);
    if (use_int_labels)
    {
        batch_labels_int.reserve(actual_batch_size);
    }
    else
    {
        batch_labels_float.reserve(actual_batch_size);
    }

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
        if (use_int_labels)
        {
            // 分类任务：统一转换为 int64，语义与 PyTorch 的 LongTensor 一致
            switch (label.dtype())
            {
                case DataType::kInt64:
                    batch_labels_int.push_back(label.item<int64_t>());
                    break;
                case DataType::kInt32:
                    batch_labels_int.push_back(static_cast<int64_t>(label.item<int32_t>()));
                    break;
                case DataType::kInt8:
                    batch_labels_int.push_back(static_cast<int64_t>(label.item<int8_t>()));
                    break;
                case DataType::kUInt8:
                    batch_labels_int.push_back(static_cast<int64_t>(label.item<uint8_t>()));
                    break;
                default:
                    // 理论上不会到这里（因为 use_int_labels 已经过滤），但为了安全加一层保护
                    batch_labels_int.push_back(static_cast<int64_t>(label.item<float>()));
                    break;
            }
        }
        else
        {
            // 非整型标签（例如回归任务），保持 float32 语义
            batch_labels_float.push_back(label.item<float>());
        }
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
    Tensor targets = use_int_labels ? Tensor(batch_labels_int, Shape{actual_batch_size}, dtype(DataType::kInt64))
                                    : Tensor(batch_labels_float, Shape{actual_batch_size}, dtype(DataType::kFloat32));

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
