#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/data/dataset.h"
#include "test_utils.h"

using namespace origin;

// 简单的测试数据集实现
class TestDataset : public Dataset
{
private:
    size_t size_;

public:
    TestDataset(size_t size) : size_(size) {}

    std::pair<Tensor, Tensor> get_item(size_t index) override
    {
        if (index >= size_)
        {
            throw std::out_of_range("Index out of range");
        }
        // 返回简单的测试数据
        auto input  = Tensor({static_cast<float>(index)}, Shape{1}, dtype(DataType::kFloat32));
        auto target = Tensor({static_cast<float>(index % 10)}, Shape{}, dtype(DataType::kFloat32));
        return std::make_pair(input, target);
    }

    size_t size() const override { return size_; }
};

class DatasetTest : public ::testing::Test
{};

TEST_F(DatasetTest, BasicInterface)
{
    TestDataset dataset(10);

    EXPECT_EQ(dataset.size(), 10U);
    EXPECT_TRUE(dataset.valid_index(0));
    EXPECT_TRUE(dataset.valid_index(9));
    EXPECT_FALSE(dataset.valid_index(10));
}

TEST_F(DatasetTest, GetItem)
{
    TestDataset dataset(5);

    auto [input, target] = dataset.get_item(0);
    EXPECT_EQ(input.shape().size(), 1U);
    EXPECT_EQ(input.shape()[0], 1U);
    EXPECT_EQ(target.shape().size(), 0U);  // 标量
}

TEST_F(DatasetTest, GetItemOutOfRange)
{
    TestDataset dataset(5);

    EXPECT_THROW(dataset.get_item(5), std::out_of_range);
}
