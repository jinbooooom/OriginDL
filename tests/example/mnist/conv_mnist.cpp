#include <iomanip>
#include "origin.h"
#include "origin/data/mnist.h"
#include "origin/data/dataloader.h"
#include "origin/optim/adam.h"
#include "origin/optim/hooks.h"
#include "origin/core/operator.h"
#include "origin/utils/metrics.h"
#include "origin/core/config.h"
#include "origin/utils/log.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/layers/conv2d.h"
#include "origin/operators/conv/max_pool2d.h"
#ifdef WITH_CUDA
#include "origin/cuda/cuda.h"
#endif

using namespace origin;

/**
 * @brief 简单的CNN模型类
 * 结构：Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear
 */
class SimpleCNN : public Module
{
private:
    // 第一层卷积：1通道 -> 64通道，3x3卷积核
    std::unique_ptr<Conv2d> conv1_;
    
    // 第二层卷积：64通道 -> 128通道，3x3卷积核
    std::unique_ptr<Conv2d> conv2_;
    
    // 全连接层1：7*7*128 -> 128
    std::unique_ptr<Linear> fc1_;
    
    // 全连接层2：128 -> 10
    std::unique_ptr<Linear> fc2_;

public:
    SimpleCNN()
        : conv1_(std::make_unique<Conv2d>(1, 64, std::make_pair(3, 3), std::make_pair(1, 1), std::make_pair(1, 1), true)),
          conv2_(std::make_unique<Conv2d>(64, 128, std::make_pair(3, 3), std::make_pair(1, 1), std::make_pair(1, 1), true)),
          fc1_(std::make_unique<Linear>(7 * 7 * 128, 128, true)),
          fc2_(std::make_unique<Linear>(128, 10, true))
    {
        // Conv2d 和 Linear 层的参数已经通过 register_parameter 注册了
        // 由于它们继承自 Layer，Layer 继承自 Module，参数会自动被 Module 收集
    }

    Tensor forward(const Tensor &input) override
    {
        // 输入形状: (N, 784) -> reshape为 (N, 1, 28, 28)
        auto x = reshape(input, Shape{input.shape()[0], 1, 28, 28});
        
        // 第一层：Conv2d(1, 64, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)
        x = conv1_->forward(x);
        x = relu(x);
        x = max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
        // 形状: (N, 64, 14, 14)
        
        // 第二层：Conv2d(64, 128, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)
        x = conv2_->forward(x);
        x = relu(x);
        x = max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
        // 形状: (N, 128, 7, 7)
        
        // Flatten: (N, 128, 7, 7) -> (N, 128*7*7) = (N, 6272)
        x = reshape(x, Shape{x.shape()[0], 128 * 7 * 7});
        
        // 全连接层1：6272 -> 128
        x = fc1_->forward(x);
        x = relu(x);
        // 形状: (N, 128)
        
        // 全连接层2：128 -> 10
        x = fc2_->forward(x);
        // 形状: (N, 10)
        
        return x;
    }

    std::vector<Parameter *> parameters() override
    {
        std::vector<Parameter *> params;

        // 首先收集当前模块自己的参数（如果有的话）
        auto base_params = Module::parameters();
        params.insert(params.end(), base_params.begin(), base_params.end());

        // 收集所有层的参数
        auto conv1_params = conv1_->parameters();
        auto conv2_params = conv2_->parameters();
        auto fc1_params = fc1_->parameters();
        auto fc2_params = fc2_->parameters();
        
        params.insert(params.end(), conv1_params.begin(), conv1_params.end());
        params.insert(params.end(), conv2_params.begin(), conv2_params.end());
        params.insert(params.end(), fc1_params.begin(), fc1_params.end());
        params.insert(params.end(), fc2_params.begin(), fc2_params.end());
        
        return params;
    }

    void to(Device device) override
    {
        // 首先迁移当前模块自己的参数（如果有的话）
        Module::to(device);
        
        // 迁移所有层到指定设备
        conv1_->to(device);
        conv2_->to(device);
        fc1_->to(device);
        fc2_->to(device);
    }
};

int main(int argc, char *argv[])
{
    // 设置随机种子
    std::srand(42);

    // 超参数
    const int max_epoch = 10;
    const int batch_size = 256;
    const float learning_rate = 0.0005f;  // 稍微降低学习率，提高稳定性
    const float weight_decay_rate = 1e-4f;
    const int log_interval = 50;  // 减少日志输出频率

    // 检测并选择设备（GPU优先，如果没有GPU则使用CPU）
    Device device(DeviceType::kCPU);
#ifdef WITH_CUDA
    if (cuda::is_available())
    {
        device = Device(DeviceType::kCUDA, 0);
        logi("CUDA is available. Using GPU for training.");
        logi("CUDA device count: {}", cuda::device_count());
    }
    else
    {
        logw("CUDA is not available. Using CPU for training.");
    }
#else
    logi("CUDA support not compiled. Using CPU for training.");
#endif

    logi("=== MNIST Handwritten Digit Recognition with CNN ===");
    logi("Device: {}", device.to_string());
    logi("Max epochs: {}", max_epoch);
    logi("Batch size: {}", batch_size);
    logi("Learning rate: {}", learning_rate);
    logi("Weight decay: {}", weight_decay_rate);
    logi("Log interval: {} (every {} batch)", log_interval, log_interval);

    // 加载数据集
    logi("Loading MNIST dataset...");
    MNIST train_dataset("./data", true);   // 训练集
    MNIST test_dataset("./data", false);   // 测试集

    logi("Train dataset size: {}", train_dataset.size());
    logi("Test dataset size: {}", test_dataset.size());

    // 创建数据加载器
    DataLoader train_loader(train_dataset, batch_size, true);   // 训练时打乱
    DataLoader test_loader(test_dataset, batch_size, false);    // 测试时不打乱

    // 创建模型
    logi("Creating CNN model...");
    SimpleCNN model;
    model.to(device);  // 将模型移到指定设备
    logi("Model created with {} parameters", model.parameters().size());

    // 创建优化器
    Adam optimizer(model, learning_rate);
    
    // 注册权重衰减Hook
    WeightDecay weight_decay(weight_decay_rate);
    optimizer.register_hook(weight_decay.hook());

    // 训练循环
    logi("Starting training...");
    for (int epoch = 0; epoch < max_epoch; ++epoch)
    {
        logi("========== Epoch {}/{} ==========", epoch + 1, max_epoch);
        
        // 训练阶段
        model.train(true);
        float train_loss = 0.0f;
        int train_batches = 0;
        int train_correct = 0;
        int train_total = 0;

        train_loader.reset();
        while (train_loader.has_next()) // 训练完整epoch
        {
            {
                auto [x, t] = train_loader.next();
                
                // 将数据移到指定设备
                x = x.to(device);
                t = t.to(device);
                
                // 前向传播
                auto y = model(x);
                auto loss = softmax_cross_entropy(y, t);
                
                // 先提取loss和准确率（在反向传播前，避免计算图累积）
                float loss_value = loss.item<float>();
                int current_batch_size = static_cast<int>(x.shape()[0]);
                
                // 计算准确率（使用no_grad避免保留计算图）
                float acc_value = 0.0f;
                {
                    auto guard = no_grad();
                    auto acc = accuracy(y, t);
                    acc_value = acc.item<float>();
                }
                
                // 反向传播
                optimizer.zero_grad();
                loss.backward();
                optimizer.step();
                
                // 显式断开计算图，释放中间tensor的内存
                if (train_batches % 20 == 0)
                {
                    loss.detach();
                    y.detach();
                }
                
                // 定期清理CUDA内存碎片
                if (train_batches % 100 == 0 && device.type() == DeviceType::kCUDA)
                {
#ifdef WITH_CUDA
                    cudaDeviceSynchronize();
#endif
                }
                
                train_correct += static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                train_total += current_batch_size;
                train_loss += loss_value;
                train_batches++;

                // 根据log_interval控制打印频率
                if (train_batches % log_interval == 0)
                {
                    float avg_loss = train_loss / train_batches;
                    float avg_acc = 100.0f * train_correct / train_total;
                    logi("Epoch {}/{} Batch {} Loss: {:.4f} Acc: {:.2f}%", 
                         epoch + 1, max_epoch, train_batches, avg_loss, avg_acc);
                }
            } // 作用域结束，自动释放x, t, y, loss, acc等tensor
        }

        float avg_train_loss = train_loss / train_batches;
        float train_acc = 100.0f * train_correct / train_total;
        
        logi("Epoch {}/{} Training Complete - Loss: {:.4f} Acc: {:.2f}%", 
             epoch + 1, max_epoch, avg_train_loss, train_acc);

        // 测试阶段
        logi("Evaluating on test set...");
        model.eval();
        float test_loss = 0.0f;
        int test_batches = 0;
        int test_correct = 0;
        int test_total = 0;

        {
            auto guard = no_grad();  // 测试时禁用梯度计算
            test_loader.reset();
            while (test_loader.has_next())
            {
                {
                    auto [x, t] = test_loader.next();
                    
                    // 将数据移到指定设备
                    x = x.to(device);
                    t = t.to(device);
                    
                    // 将 targets 从 float 转换为 int32_t 类型（softmax_cross_entropy 需要）
                    auto t_float_data = t.to(Device(DeviceType::kCPU)).to_vector<float>();
                    std::vector<int32_t> t_int32_data(t_float_data.size());
                    for (size_t i = 0; i < t_float_data.size(); ++i)
                    {
                        t_int32_data[i] = static_cast<int32_t>(t_float_data[i]);
                    }
                    auto t_int32 = Tensor(t_int32_data, t.shape(), dtype(DataType::kInt32).device(device));
                    
                    // 前向传播（不需要梯度，已在no_grad作用域内）
                    auto y = model(x);
                    auto loss = softmax_cross_entropy(y, t_int32);
                    float loss_val = loss.item<float>();
                    
                    // 使用accuracy函数计算准确率
                    auto acc = accuracy(y, t_int32);
                    float acc_value = acc.item<float>();
                    int current_batch_size = static_cast<int>(x.shape()[0]);
                    
                    int batch_correct = static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                    test_correct += batch_correct;
                    test_total += current_batch_size;
                    test_loss += loss_val;
                    test_batches++;
                    
                    // 根据log_interval控制打印频率
                    if (test_batches % log_interval == 0)
                    {
                        float avg_test_loss_so_far = test_loss / test_batches;
                        float avg_test_acc_so_far = 100.0f * test_correct / test_total;
                        logi("Test Batch {} Loss: {:.4f} Acc: {:.2f}%", 
                             test_batches, avg_test_loss_so_far, avg_test_acc_so_far);
                    }
                } // 作用域结束，自动释放x, t, t_int32, y, loss, acc等tensor
            }
        }

        float avg_test_loss = test_loss / test_batches;
        float test_acc = 100.0f * test_correct / test_total;

        // 输出epoch结果
        logi("========== Epoch {}/{} Summary ==========", epoch + 1, max_epoch);
        logi("  Train Loss: {:.4f}, Train Acc: {:.2f}%", avg_train_loss, train_acc);
        logi("  Test Loss:  {:.4f}, Test Acc:  {:.2f}%", avg_test_loss, test_acc);
        logi("===========================================");
    }

    logi("Training completed!");
    return 0;
}

