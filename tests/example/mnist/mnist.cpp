#include <iomanip>
#include "origin.h"
#include "origin/nn/models/mlp.h"
#include "origin/data/mnist.h"
#include "origin/data/dataloader.h"
#include "origin/optim/adam.h"
#include "origin/optim/hooks.h"
#include "origin/core/operator.h"
#include "origin/utils/metrics.h"
#include "origin/core/config.h"
#include "origin/utils/log.h"
#ifdef WITH_CUDA
#include "origin/cuda/cuda.h"
#endif

using namespace origin;

int main(int argc, char *argv[])
{
    // 设置随机种子（可选）
    std::srand(42);

    // 超参数
    const int max_epoch = 5;
    const int batch_size = 100;
    const int hidden_size = 1000;
    const float learning_rate = 0.001f;
    const float weight_decay_rate = 1e-4f;
    
    // 日志打印频率控制（设置为1表示每个batch都打印，用于调试）
    const int log_interval = 1;  // 可以修改为10、50等来减少打印频率

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

    logi("=== MNIST Handwritten Digit Recognition Demo ===");
    logi("Device: {}", device.to_string());
    logi("Max epochs: {}", max_epoch);
    logi("Batch size: {}", batch_size);
    logi("Hidden size: {}", hidden_size);
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
    logi("Creating MLP model...");
    MLP model({784, hidden_size, hidden_size, 10});  // 输入784维，两个隐藏层，输出10维
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
        while (train_loader.has_next())
        {
            auto [x, t] = train_loader.next();
            
            // 将数据移到指定设备
            x = x.to(device);
            t = t.to(device);
            
            // 前向传播
            auto y = model(x);
            auto loss = softmax_cross_entropy(y, t);
            
            // 反向传播
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            // 计算准确率
            auto acc = accuracy(y, t);
            train_correct += static_cast<int>(acc.item<float>() * static_cast<float>(x.shape()[0]));
            train_total += static_cast<int>(x.shape()[0]);
            train_loss += loss.item<float>();
            train_batches++;

            // 根据log_interval控制打印频率
            if (train_batches % log_interval == 0)
            {
                float avg_loss = train_loss / train_batches;
                float avg_acc = 100.0f * train_correct / train_total;
                logi("Epoch {}/{} Batch {} Loss: {:.4f} Acc: {:.2f}%", 
                     epoch + 1, max_epoch, train_batches, avg_loss, avg_acc);
            }
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
                auto [x, t] = test_loader.next();
                
                // 将数据移到指定设备
                x = x.to(device);
                t = t.to(device);
                
                // 将 targets 从 float 转换为 int32_t 类型（softmax_cross_entropy 需要）
                auto t_float_data = t.to_vector<float>();
                std::vector<int32_t> t_int32_data(t_float_data.size());
                for (size_t i = 0; i < t_float_data.size(); ++i)
                {
                    t_int32_data[i] = static_cast<int32_t>(t_float_data[i]);
                }
                auto t_int32 = Tensor(t_int32_data, t.shape(), dtype(DataType::kInt32).device(device));
                
                // 前向传播（不需要梯度）
                auto y = model(x);
                auto loss = softmax_cross_entropy(y, t_int32);
                float loss_val = loss.item<float>();
                
                // 计算准确率
                auto y_data = y.to_vector<float>();
                size_t batch_size = x.shape()[0];
                size_t num_classes = y.shape()[1];
                
                int batch_correct = 0;
                for (size_t i = 0; i < batch_size; ++i)
                {
                    // 找到预测类别（argmax）
                    size_t pred_class = 0;
                    float max_val = y_data[i * num_classes];
                    for (size_t j = 1; j < num_classes; ++j)
                    {
                        if (y_data[i * num_classes + j] > max_val)
                        {
                            max_val = y_data[i * num_classes + j];
                            pred_class = j;
                        }
                    }
                    
                    // 检查是否正确
                    if (static_cast<size_t>(t_int32_data[i]) == pred_class)
                    {
                        test_correct++;
                        batch_correct++;
                    }
                }
                test_total += static_cast<int>(batch_size);
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

