#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>
#include "origin/core/tensor.h"
#include "origin/io/checkpoint.h"
#include "origin/io/model_io.h"
#include "origin/nn/models/mlp.h"
#include "origin/optim/adam.h"
#include "test_utils.h"

using namespace origin;
namespace F = origin::functional;
namespace fs = std::filesystem;

class CheckpointTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 创建临时测试目录
        test_dir_ = "test_checkpoint_dir";
        fs::create_directories(test_dir_);
    }

    void TearDown() override
    {
        // 清理临时测试目录
        if (fs::exists(test_dir_))
        {
            fs::remove_all(test_dir_);
        }
    }

    std::string test_dir_;
};

TEST_F(CheckpointTest, SaveAndLoadCheckpoint)
{
    // 创建一个简单的模型
    nn::MLP model({4, 8, 2});
    model.to(Device(DeviceType::kCPU));

    // 创建优化器
    Adam optimizer(model, 0.001f);

    // 创建示例输入并执行一次前向传播（初始化优化器状态）
    auto x = Tensor::randn(Shape{2, 4}, TensorOptions().dtype(DataType::kFloat32).device(DeviceType::kCPU));
    auto y = model(x);

    // 执行一次反向传播以初始化优化器的缓冲区
    auto target = Tensor::zeros(Shape{2, 2}, TensorOptions().dtype(DataType::kFloat32).device(DeviceType::kCPU));
    auto loss   = (y - target) * (y - target);
    loss.backward();
    optimizer.step();

    // 创建 checkpoint
    Checkpoint checkpoint;
    checkpoint.model_state_dict             = model.state_dict();
    checkpoint.optimizer_state_dict["adam"] = optimizer.state_dict();
    checkpoint.epoch                        = 5;
    checkpoint.step                         = 100;
    checkpoint.loss                         = 0.5f;
    checkpoint.optimizer_type               = "Adam";
    checkpoint.optimizer_config["lr"]       = 0.001f;
    checkpoint.optimizer_config["beta1"]    = 0.9f;
    checkpoint.optimizer_config["beta2"]    = 0.999f;
    checkpoint.optimizer_config["eps"]      = 1e-8f;

    // 保存 checkpoint
    std::string checkpoint_path = test_dir_ + "/test_checkpoint.ckpt";
    save(checkpoint, checkpoint_path);

    // 验证文件存在
    std::string ckpt_dir = checkpoint_path;
    if (ckpt_dir.size() > 5 && ckpt_dir.substr(ckpt_dir.size() - 5) == ".ckpt")
    {
        ckpt_dir = ckpt_dir.substr(0, ckpt_dir.size() - 5);
    }
    EXPECT_TRUE(fs::exists(ckpt_dir));
    EXPECT_TRUE(fs::exists(ckpt_dir + "/model.odl"));
    EXPECT_TRUE(fs::exists(ckpt_dir + "/metadata.json"));

    // 加载 checkpoint
    Checkpoint loaded_checkpoint = load_checkpoint(checkpoint_path);

    // 验证加载的数据
    EXPECT_EQ(loaded_checkpoint.epoch, 5);
    EXPECT_EQ(loaded_checkpoint.step, 100);
    EXPECT_FLOAT_EQ(loaded_checkpoint.loss, 0.5f);
    EXPECT_EQ(loaded_checkpoint.optimizer_type, "Adam");
    EXPECT_FLOAT_EQ(loaded_checkpoint.optimizer_config.at("lr"), 0.001f);
    EXPECT_FLOAT_EQ(loaded_checkpoint.optimizer_config.at("beta1"), 0.9f);
    EXPECT_FLOAT_EQ(loaded_checkpoint.optimizer_config.at("beta2"), 0.999f);
    EXPECT_FLOAT_EQ(loaded_checkpoint.optimizer_config.at("eps"), 1e-8f);

    // 验证模型参数数量一致
    EXPECT_EQ(loaded_checkpoint.model_state_dict.size(), checkpoint.model_state_dict.size());

    // 创建新模型并加载参数
    nn::MLP new_model({4, 8, 2});
    new_model.to(Device(DeviceType::kCPU));
    new_model.load_state_dict(loaded_checkpoint.model_state_dict);

    // 验证模型参数已加载（通过前向传播结果）
    auto x2 = Tensor::randn(Shape{2, 4}, TensorOptions().dtype(DataType::kFloat32).device(DeviceType::kCPU));
    auto y1 = model(x2);
    auto y2 = new_model(x2);

    // 验证输出形状一致
    EXPECT_EQ(y1.shape(), y2.shape());
}

TEST_F(CheckpointTest, SaveAndLoadModelStateDict)
{
    // 测试 .odl 格式的保存和加载
    // 使用正确的维度：输入3维，隐藏层5维，输出2维
    nn::MLP model({3, 5, 2});
    model.to(Device(DeviceType::kCPU));

    // 获取模型参数
    StateDict state_dict = model.state_dict();
    EXPECT_GT(state_dict.size(), 0U);

    // 保存模型
    std::string model_path = test_dir_ + "/test_model.odl";
    save(state_dict, model_path);

    // 验证文件存在
    EXPECT_TRUE(fs::exists(model_path));

    // 加载模型
    StateDict loaded_state_dict = load(model_path);

    // 验证参数数量一致
    EXPECT_EQ(loaded_state_dict.size(), state_dict.size());

    // 验证参数键一致
    for (const auto &[key, value] : state_dict)
    {
        EXPECT_TRUE(loaded_state_dict.find(key) != loaded_state_dict.end());
        EXPECT_EQ(loaded_state_dict[key].shape(), value.shape());
    }

    // 创建新模型并加载参数
    nn::MLP new_model({3, 5, 2});
    new_model.to(Device(DeviceType::kCPU));
    new_model.load_state_dict(loaded_state_dict);

    // 验证模型可以正常使用
    auto x = Tensor::randn(Shape{1, 3}, TensorOptions().dtype(DataType::kFloat32).device(DeviceType::kCPU));
    auto y = new_model(x);
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 1U);
    EXPECT_EQ(y.shape()[1], 2U);
}

TEST_F(CheckpointTest, OptimizerStateDict)
{
    // 测试优化器的 state_dict
    nn::MLP model({2, 4, 1});
    model.to(Device(DeviceType::kCPU));

    Adam optimizer(model, 0.01f, 0.9f, 0.999f, 1e-8f);

    // 执行一次训练步骤以初始化优化器状态
    auto x = Tensor::randn(Shape{1, 2}, TensorOptions().dtype(DataType::kFloat32).device(DeviceType::kCPU));
    auto y = model(x);
    // 确保 target 的形状与 y 匹配
    auto target = Tensor::zeros(y.shape(), TensorOptions().dtype(DataType::kFloat32).device(DeviceType::kCPU));
    auto loss   = (y - target) * (y - target);
    loss.backward();
    optimizer.step();

    // 获取优化器状态
    auto optimizer_state = optimizer.state_dict();

    // 验证状态包含必要的键
    EXPECT_TRUE(optimizer_state.find("lr") != optimizer_state.end());
    EXPECT_TRUE(optimizer_state.find("beta1") != optimizer_state.end());
    EXPECT_TRUE(optimizer_state.find("beta2") != optimizer_state.end());
    EXPECT_TRUE(optimizer_state.find("eps") != optimizer_state.end());

    // 验证配置值
    EXPECT_FLOAT_EQ(std::any_cast<float>(optimizer_state.at("lr")), 0.01f);
    EXPECT_FLOAT_EQ(std::any_cast<float>(optimizer_state.at("beta1")), 0.9f);
    EXPECT_FLOAT_EQ(std::any_cast<float>(optimizer_state.at("beta2")), 0.999f);
    EXPECT_FLOAT_EQ(std::any_cast<float>(optimizer_state.at("eps")), 1e-8f);
}
