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
#include "origin/operators/conv/conv2d.h"
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
    Parameter W1_;  // (64, 1, 3, 3)
    Parameter b1_;  // (64,)
    
    // 第二层卷积：64通道 -> 128通道，3x3卷积核
    Parameter W2_;  // (128, 64, 3, 3)
    Parameter b2_;  // (128,)
    
    // 全连接层1：7*7*128 -> 128
    std::unique_ptr<Linear> fc1_;
    
    // 全连接层2：128 -> 10
    std::unique_ptr<Linear> fc2_;

public:
    SimpleCNN()
        : W1_(Tensor::randn(Shape{64, 1, 3, 3}, TensorOptions(DataType::kFloat32))),
          b1_(Tensor::zeros(Shape{64}, TensorOptions(DataType::kFloat32))),
          W2_(Tensor::randn(Shape{128, 64, 3, 3}, TensorOptions(DataType::kFloat32))),
          b2_(Tensor::zeros(Shape{128}, TensorOptions(DataType::kFloat32))),
          fc1_(std::make_unique<Linear>(7 * 7 * 128, 128, true)),
          fc2_(std::make_unique<Linear>(128, 10, true))
    {
        // 初始化卷积核权重（使用Kaiming初始化，更适合ReLU）
        initialize_weights(W1_, 64 * 1 * 3 * 3);
        initialize_weights(W2_, 128 * 64 * 3 * 3);
        
        // 注册参数，以便优化器能够访问
        register_parameter("W1", W1_);
        register_parameter("b1", b1_);
        register_parameter("W2", W2_);
        register_parameter("b2", b2_);
    }

    Tensor forward(const Tensor &input) override
    {
        // 输入形状: (N, 784) -> reshape为 (N, 1, 28, 28)
        auto x = reshape(input, Shape{input.shape()[0], 1, 28, 28});
        
        // 第一层：Conv2d(1, 64, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)
        x = conv2d(x, W1_, &b1_, {1, 1}, {1, 1});
        x = relu(x);
        x = max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
        // 形状: (N, 64, 14, 14)
        
        // 第二层：Conv2d(64, 128, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)
        x = conv2d(x, W2_, &b2_, {1, 1}, {1, 1});
        x = relu(x);
        x = max_pool2d(x, {2, 2}, {2, 2}, {0, 0});
        // 形状: (N, 128, 7, 7)
        
        // Flatten: (N, 128, 7, 7) -> (N, 128*7*7) = (N, 6272)
        x = reshape(x, Shape{x.shape()[0], 128 * 7 * 7});
        
        // 全连接层1：3136 -> 128
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
        params.push_back(&W1_);
        params.push_back(&b1_);
        params.push_back(&W2_);
        params.push_back(&b2_);
        
        // 添加全连接层的参数
        auto fc1_params = fc1_->parameters();
        auto fc2_params = fc2_->parameters();
        params.insert(params.end(), fc1_params.begin(), fc1_params.end());
        params.insert(params.end(), fc2_params.begin(), fc2_params.end());
        
        return params;
    }

    void to(Device device) override
    {
        // 先调用基类的to()方法，迁移所有注册的参数
        Module::to(device);
        
        // 然后确保所有参数都在正确的设备上
        // 注意：Module::to()已经迁移了注册的参数，但我们需要更新成员变量
        W1_ = Parameter(W1_.to(device));
        b1_ = Parameter(b1_.to(device));
        W2_ = Parameter(W2_.to(device));
        b2_ = Parameter(b2_.to(device));
        
        // 子模块的to()已经在Module::to()中调用，但为了确保，再次调用
        fc1_->to(device);
        fc2_->to(device);
    }

private:
    void initialize_weights(Parameter &param, int fan_in)
    {
        // Kaiming初始化（He初始化）：更适合ReLU激活函数
        // 对于ReLU，使用 std::sqrt(2.0 / fan_in)
        // 使用正态分布：N(0, sqrt(2/fan_in))
        float std_dev = std::sqrt(2.0f / static_cast<float>(fan_in));
        auto device = param.device();  // 保存设备信息
        auto data = param.to_vector<float>();
        
        // 使用Box-Muller变换生成正态分布随机数
        for (size_t i = 0; i < data.size(); ++i)
        {
            // 生成两个均匀分布随机数
            float u1 = static_cast<float>(std::rand()) / RAND_MAX;
            float u2 = static_cast<float>(std::rand()) / RAND_MAX;
            // Box-Muller变换：从均匀分布生成标准正态分布
            float z = std::sqrt(-2.0f * std::log(u1 + 1e-8f)) * std::cos(2.0f * 3.14159265359f * u2);
            // 应用标准差
            data[i] = z * std_dev;
        }
        // 创建新Tensor时保持设备信息
        param = Tensor(data, param.shape(), dtype(DataType::kFloat32).device(device));
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

