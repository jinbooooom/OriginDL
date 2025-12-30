#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/data/dataloader.h"
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

class DataLoaderTest : public ::testing::Test
{};

TEST_F(DataLoaderTest, BasicIteration)
{
    TestDataset dataset(10);
    DataLoader loader(dataset, 3, false);  // batch_size=3, shuffle=false

    EXPECT_TRUE(loader.has_next());
    EXPECT_EQ(loader.batch_size(), 3U);
    EXPECT_EQ(loader.dataset_size(), 10U);

    // 获取第一个批次
    auto [inputs, targets] = loader.next();
    EXPECT_EQ(inputs.shape()[0], 3U);
    EXPECT_EQ(targets.shape()[0], 3U);

    // 获取第二个批次
    auto [inputs2, targets2] = loader.next();
    EXPECT_EQ(inputs2.shape()[0], 3U);
    EXPECT_EQ(targets2.shape()[0], 3U);

    // 获取第三个批次（不完整）
    auto [inputs3, targets3] = loader.next();
    EXPECT_EQ(inputs3.shape()[0], 3U);
    EXPECT_EQ(targets3.shape()[0], 3U);

    // 获取第四个批次（最后一个，不完整）
    auto [inputs4, targets4] = loader.next();
    EXPECT_EQ(inputs4.shape()[0], 1U);
    EXPECT_EQ(targets4.shape()[0], 1U);

    // 应该没有更多数据了
    EXPECT_FALSE(loader.has_next());
}

TEST_F(DataLoaderTest, Reset)
{
    TestDataset dataset(5);
    DataLoader loader(dataset, 2, false);

    // 迭代一次
    loader.next();
    EXPECT_TRUE(loader.has_next());

    // 重置
    loader.reset();
    EXPECT_TRUE(loader.has_next());

    // 应该可以重新迭代
    auto [inputs, targets] = loader.next();
    EXPECT_EQ(inputs.shape()[0], 2U);
}

TEST_F(DataLoaderTest, Shuffle)
{
    TestDataset dataset(10);
    DataLoader loader1(dataset, 10, false);  // 不打乱
    DataLoader loader2(dataset, 10, true);   // 打乱

    // 获取两个批次（应该不同，因为打乱了）
    auto [inputs1, targets1] = loader1.next();
    auto [inputs2, targets2] = loader2.next();

    // 由于随机性，我们不能保证一定不同，但至少应该能正常运行
    EXPECT_EQ(inputs1.shape()[0], 10U);
    EXPECT_EQ(inputs2.shape()[0], 10U);
}

TEST_F(DataLoaderTest, BatchSizeOne)
{
    TestDataset dataset(5);
    DataLoader loader(dataset, 1, false);

    for (size_t i = 0; i < 5; ++i)
    {
        EXPECT_TRUE(loader.has_next());
        auto [inputs, targets] = loader.next();
        EXPECT_EQ(inputs.shape()[0], 1U);
        EXPECT_EQ(targets.shape()[0], 1U);
    }

    EXPECT_FALSE(loader.has_next());
}

TEST_F(DataLoaderTest, EmptyDataset)
{
    TestDataset dataset(0);
    DataLoader loader(dataset, 1, false);

    EXPECT_FALSE(loader.has_next());
    EXPECT_EQ(loader.dataset_size(), 0U);
    // 对于空数据集，调用 next() 应该抛出异常
    EXPECT_THROW(loader.next(), std::runtime_error);
}
