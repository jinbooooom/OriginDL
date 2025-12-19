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
    const int max_epoch = 3;
    const int batch_size = 64;  // 减小batch size以降低GPU内存使用
    const int hidden_size = 256;  // 减小hidden size以降低GPU内存使用，调大精度更高
    const float learning_rate = 0.001f;
    const float weight_decay_rate = 1e-4f;
    
    // 日志打印频率控制（设置为1表示每个batch都打印，用于调试）
    const int log_interval = 1;  // 可以修改为10、50等来减少打印频率

    // 检测并选择设备（GPU优先，如果没有GPU则使用CPU）
    Device device(DeviceType::kCPU);
#ifdef WITH_CUDA
    // if (0 &&cuda::is_available())// 强行使用cpu
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
        while (train_batches < 30 && train_loader.has_next()) // 为了加快速度，所以只训练30个batch
        // while (train_loader.has_next())
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
                // 使用detach()方法断开tensor与计算图的连接，帮助释放GPU内存
                // 注意：这里只断开loss和y，因为它们是计算图的根节点
                // 断开后，整个计算图（包括所有中间tensor）都可以被释放
                // 为了减少性能开销，每50个batch才detach一次（减少递归遍历的开销）
                if (train_batches % 20 == 0)
                {
                    loss.detach();
                    y.detach();
                }
                
                // 定期清理CUDA内存碎片（每100个batch清理一次）
                // 这可以帮助释放未使用的GPU内存，避免内存碎片化导致训练变慢
                if (train_batches % 100 == 0 && device.type() == DeviceType::kCUDA)
                {
#ifdef WITH_CUDA
                    cudaDeviceSynchronize();  // 确保所有CUDA操作完成
                    // 注意：CUDA没有像PyTorch的empty_cache()那样的函数
                    // 但cudaDeviceSynchronize()可以帮助释放一些内存
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
                    // 在CPU上转换，避免GPU内存占用
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
                    
                    // 使用accuracy函数计算准确率（更高效，避免多次to_vector）
                    auto acc = accuracy(y, t_int32);
                    float acc_value = acc.item<float>();
                    int current_batch_size = static_cast<int>(x.shape()[0]);
                    
                    // 显式断开计算图（虽然已经在no_grad中，但为了确保内存释放）
                    // 通过作用域和变量重新赋值来帮助释放
                    
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

/*
。。。
node-worker-147 2025-12-19 10:09:51.428 I 162309 162309 [mnist.cpp:main:239] ========== Epoch 3/3 Summary ==========
node-worker-147 2025-12-19 10:09:51.428 I 162309 162309 [mnist.cpp:main:240]   Train Loss: 0.3374, Train Acc: 90.47%
node-worker-147 2025-12-19 10:09:51.428 I 162309 162309 [mnist.cpp:main:241]   Test Loss:  0.3408, Test Acc:  89.75%
node-worker-147 2025-12-19 10:09:51.428 I 162309 162309 [mnist.cpp:main:242] ===========================================
node-worker-147 2025-12-19 10:09:51.428 I 162309 162309 [mnist.cpp:main:245] Training completed!
*/

