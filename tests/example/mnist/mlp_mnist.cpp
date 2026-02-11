#include <getopt.h>
#include <iomanip>
#include "origin.h"
#include "origin/nn/models/mlp.h"

using namespace origin;
namespace F  = origin::functional;
namespace nn = origin::nn;

struct TrainingConfig
{
    int max_epoch           = 10;
    int batch_size          = 256;
    int hidden_size         = 1000;
    float learning_rate     = 0.001f;
    float weight_decay_rate = 1e-4f;
    int log_interval        = 50;
    std::string model_path  = "model/mnist_mlp_model.odl";
    int checkpoint_interval = 5;
    int random_seed         = 42;
    std::string data_dir    = "./data/mnist";
    int device_id           = -2;  // -2=auto, -1=CPU, >=0=GPU id

    std::string checkpoint_dir() const
    {
        size_t last_slash = model_path.find_last_of('/');
        if (last_slash != std::string::npos)
        {
            return model_path.substr(0, last_slash + 1) + "checkpoints";
        }
        return "checkpoints";
    }

    void print() const
    {
        logi("=== Training Configuration ===");
        logi("Max epochs: {}", max_epoch);
        logi("Batch size: {}", batch_size);
        logi("Hidden size: {}", hidden_size);
        logi("Learning rate: {}", learning_rate);
        logi("Weight decay: {}", weight_decay_rate);
        logi("Log interval: {}", log_interval);
        logi("Model path: {}", model_path);
        logi("Checkpoint dir: {}", checkpoint_dir());
        logi("Checkpoint interval: {} epochs", checkpoint_interval);
        logi("Random seed: {}", random_seed);
        logi("Data dir: {}", data_dir);
        logi("Device id: {} (-2=auto -1=CPU >=0=GPU)", device_id);
        logi("==============================");
    }
};

static void usage(const char *program_name)
{
    loga("Usage: {} [OPTIONS]\n", program_name);
    loga("Options:\n");
    loga("  -e, --epochs EPOCHS          Maximum number of epochs (default: 10)\n");
    loga("  -b, --batch-size SIZE        Batch size (default: 256)\n");
    loga("  -H, --hidden-size SIZE       Hidden layer size (default: 1000)\n");
    loga("  -l, --learning-rate LR       Learning rate (default: 0.001)\n");
    loga("  -w, --weight-decay RATE      Weight decay rate (default: 1e-4)\n");
    loga("  -i, --log-interval INTERVAL  Log interval in batches (default: 50)\n");
    loga("  -m, --model-path PATH        Path to save model (default: model/mnist_mlp_model.odl)\n");
    loga("  -c, --checkpoint-interval N  Save checkpoint every N epochs (default: 5)\n");
    loga("  -s, --seed SEED              Random seed (default: 42)\n");
    loga("  -p, --path DIR               MNIST data directory (default: ./data/mnist)\n");
    loga("  -d, --device ID              Device: -2=auto, -1=CPU, >=0=GPU id (default: auto)\n");
    loga("  -h, --help                   Show this help message\n");
}

static TrainingConfig parse_args(int argc, char *argv[])
{
    TrainingConfig config;

    static struct option long_options[] = {{"epochs", required_argument, 0, 'e'},
                                           {"batch-size", required_argument, 0, 'b'},
                                           {"hidden-size", required_argument, 0, 'H'},
                                           {"learning-rate", required_argument, 0, 'l'},
                                           {"weight-decay", required_argument, 0, 'w'},
                                           {"log-interval", required_argument, 0, 'i'},
                                           {"model-path", required_argument, 0, 'm'},
                                           {"checkpoint-interval", required_argument, 0, 'c'},
                                           {"seed", required_argument, 0, 's'},
                                           {"path", required_argument, 0, 'p'},
                                           {"device", required_argument, 0, 'd'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int c;
    while ((c = getopt_long(argc, argv, "e:b:H:l:w:i:m:c:s:p:d:h", long_options, &option_index)) != -1)
    {
        switch (c)
        {
            case 'e':
                config.max_epoch = std::atoi(optarg);
                if (config.max_epoch <= 0)
                {
                    logw("Invalid max_epoch: {}. Using default: 10", optarg);
                    config.max_epoch = 10;
                }
                break;
            case 'b':
                config.batch_size = std::atoi(optarg);
                if (config.batch_size <= 0)
                {
                    logw("Invalid batch_size: {}. Using default: 256", optarg);
                    config.batch_size = 256;
                }
                break;
            case 'H':
                config.hidden_size = std::atoi(optarg);
                if (config.hidden_size <= 0)
                {
                    logw("Invalid hidden_size: {}. Using default: 1000", optarg);
                    config.hidden_size = 1000;
                }
                break;
            case 'l':
                config.learning_rate = std::atof(optarg);
                if (config.learning_rate <= 0.0f)
                {
                    logw("Invalid learning_rate: {}. Using default: 0.001", optarg);
                    config.learning_rate = 0.001f;
                }
                break;
            case 'w':
                config.weight_decay_rate = std::atof(optarg);
                if (config.weight_decay_rate < 0.0f)
                {
                    logw("Invalid weight_decay_rate: {}. Using default: 1e-4", optarg);
                    config.weight_decay_rate = 1e-4f;
                }
                break;
            case 'i':
                config.log_interval = std::atoi(optarg);
                if (config.log_interval <= 0)
                {
                    logw("Invalid log_interval: {}. Using default: 50", optarg);
                    config.log_interval = 50;
                }
                break;
            case 'm':
                config.model_path = optarg;
                break;
            case 'c':
                config.checkpoint_interval = std::atoi(optarg);
                if (config.checkpoint_interval <= 0)
                {
                    logw("Invalid checkpoint_interval: {}. Using default: 5", optarg);
                    config.checkpoint_interval = 5;
                }
                break;
            case 's':
                config.random_seed = std::atoi(optarg);
                break;
            case 'p':
                config.data_dir = optarg;
                break;
            case 'd':
                config.device_id = std::atoi(optarg);
                break;
            case 'h':
                usage(argv[0]);
                std::exit(0);
            case '?':
                loga("Use -h or --help for usage information");
                std::exit(1);
            default:
                break;
        }
    }
    return config;
}

int main(int argc, char *argv[])
{
    TrainingConfig config = parse_args(argc, argv);

    std::srand(config.random_seed);

    int device_id = config.device_id;
    if (device_id == -2)
    {
        device_id = cuda::is_available() ? 0 : -1;
    }

    Device device(DeviceType::kCPU);
    bool use_gpu = (device_id >= 0);
    if (use_gpu)
    {
        if (!cuda::is_available())
        {
            loge("CUDA is not available on this system.");
            return 1;
        }
        int device_count = cuda::device_count();
        if (device_id >= device_count)
        {
            loge("Invalid GPU device ID: {}. Available devices: 0-{}", device_id, device_count - 1);
            return 1;
        }
        device = Device(DeviceType::kCUDA, device_id);
        cuda::set_device(device_id);
        logi("Device: {}", device.to_string());
        cuda::device_info();
    }
    else
    {
        logi("Device: {}", device.to_string());
    }

    logi("=== MNIST Handwritten Digit Recognition Demo (MLP) ===");
    config.print();

    logi("Loading MNIST dataset...");
    MNIST train_dataset(config.data_dir, true);
    MNIST test_dataset(config.data_dir, false);

    logi("Train dataset size: {}", train_dataset.size());
    logi("Test dataset size: {}", test_dataset.size());

    DataLoader train_loader(train_dataset, config.batch_size, true);
    DataLoader test_loader(test_dataset, config.batch_size, false);

    logi("Creating MLP model...");
    nn::MLP model({784, config.hidden_size, config.hidden_size, 10});
    model.to(device);
    logi("Model created with {} parameters", model.parameters().size());

    Adam optimizer(model, config.learning_rate);
    WeightDecay weight_decay(config.weight_decay_rate);
    optimizer.register_hook(weight_decay.hook());

    logi("Starting training...");
    for (int epoch = 0; epoch < config.max_epoch; ++epoch)
    {
        logi("========== Epoch {}/{} ==========", epoch + 1, config.max_epoch);

        // 训练阶段
        model.train(true);
        float train_loss  = 0.0f;
        int train_batches = 0;
        int train_correct = 0;
        int train_total   = 0;

        // 在每个 epoch 开始时调用 reset()，清空并重新生成索引列表, 随机打乱索引。
        // 确保每个 epoch 都能遍历完整的训练集，且数据顺序不同，有助于模型训练。
        train_loader.reset();
        // while (train_batches < 30 && train_loader.has_next()) // 为了加快速度，所以只训练30个batch
        while (train_loader.has_next())
        {
            {
                auto [x, t] = train_loader.next();

                // 将数据移到指定设备
                x = x.to(device);
                t = t.to(device);

                // 前向传播
                auto y    = model(x);
                auto loss = F::softmax_cross_entropy(y, t);

                // 先提取loss和准确率（在反向传播前，避免计算图累积）
                float loss_value       = loss.item<float>();
                int current_batch_size = static_cast<int>(x.shape()[0]);

                // 计算准确率（使用no_grad避免保留计算图）
                float acc_value = 0.0f;
                {
                    auto guard = no_grad();  // 类似 torch.no_grad()，在作用域内禁用梯度计算
                    auto acc   = accuracy(y, t);
                    acc_value  = acc.item<float>();
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
                    cuda::synchronize();  // 确保所有CUDA操作完成
                                          // 注意：CUDA没有像PyTorch的empty_cache()那样的函数
                                          // 但synchronize()可以帮助释放一些内存
                }

                train_correct += static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                train_total += current_batch_size;
                train_loss += loss_value;
                train_batches++;

                if (train_batches % config.log_interval == 0)
                {
                    float avg_loss = train_loss / train_batches;
                    float avg_acc  = 100.0f * train_correct / train_total;
                    logi("Epoch {}/{} Batch {} Loss: {:.4f} Acc: {:.2f}%", epoch + 1, config.max_epoch, train_batches,
                         avg_loss, avg_acc);
                }
            }  // 作用域结束，自动释放x, t, y, loss, acc等tensor
        }

        float avg_train_loss = train_loss / train_batches;
        float train_acc      = 100.0f * train_correct / train_total;

        logi("Epoch {}/{} Training Complete - Loss: {:.4f} Acc: {:.2f}%", epoch + 1, config.max_epoch, avg_train_loss,
             train_acc);

        // 测试阶段
        logi("Evaluating on test set...");
        model.eval();
        float test_loss  = 0.0f;
        int test_batches = 0;
        int test_correct = 0;
        int test_total   = 0;

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
                    auto y         = model(x);
                    auto loss      = F::softmax_cross_entropy(y, t_int32);
                    float loss_val = loss.item<float>();

                    // 使用accuracy函数计算准确率（更高效，避免多次to_vector）
                    auto acc               = accuracy(y, t_int32);
                    float acc_value        = acc.item<float>();
                    int current_batch_size = static_cast<int>(x.shape()[0]);

                    // 显式断开计算图（虽然已经在no_grad中，但为了确保内存释放）
                    // 通过作用域和变量重新赋值来帮助释放

                    int batch_correct = static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                    test_correct += batch_correct;
                    test_total += current_batch_size;
                    test_loss += loss_val;
                    test_batches++;

                    if (test_batches % config.log_interval == 0)
                    {
                        float avg_test_loss_so_far = test_loss / test_batches;
                        float avg_test_acc_so_far  = 100.0f * test_correct / test_total;
                        logi("Test Batch {} Loss: {:.4f} Acc: {:.2f}%", test_batches, avg_test_loss_so_far,
                             avg_test_acc_so_far);
                    }
                }  // 作用域结束，自动释放x, t, t_int32, y, loss, acc等tensor
            }
        }

        float avg_test_loss = test_loss / test_batches;
        float test_acc      = 100.0f * test_correct / test_total;

        logi("========== Epoch {}/{} Summary ==========", epoch + 1, config.max_epoch);
        logi("  Train Loss: {:.4f}, Train Acc: {:.2f}%", avg_train_loss, train_acc);
        logi("  Test Loss:  {:.4f}, Test Acc:  {:.2f}%", avg_test_loss, test_acc);
        logi("===========================================");

        if ((epoch + 1) % config.checkpoint_interval == 0 || (epoch + 1) == config.max_epoch)
        {
            try
            {
                std::string ckpt_dir = config.checkpoint_dir();
                (void)system(("mkdir -p " + ckpt_dir).c_str());
                std::string checkpoint_path = ckpt_dir + "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".ckpt";

                Checkpoint checkpoint;
                checkpoint.model_state_dict             = model.state_dict();
                checkpoint.optimizer_state_dict["adam"] = optimizer.state_dict();
                checkpoint.epoch                        = epoch + 1;
                checkpoint.step                         = train_batches;
                checkpoint.loss                         = avg_test_loss;
                checkpoint.optimizer_type               = "Adam";
                checkpoint.optimizer_config["lr"]       = config.learning_rate;
                checkpoint.optimizer_config["beta1"]    = 0.9f;
                checkpoint.optimizer_config["beta2"]    = 0.999f;
                checkpoint.optimizer_config["eps"]      = 1e-8f;

                save(checkpoint, checkpoint_path);
                logi("Checkpoint saved to {}", checkpoint_path);
            }
            catch (const std::exception &e)
            {
                logw("Failed to save checkpoint: {}", e.what());
            }
        }
    }

    logi("Training completed!");

    logi("Saving model to {}...", config.model_path);
    try
    {
        (void)system("mkdir -p model");
        model.eval();
        save(model.state_dict(), config.model_path);
        logi("Model saved successfully to {}", config.model_path);
    }
    catch (const std::exception &e)
    {
        logw("Failed to save model: {}", e.what());
        return 1;
    }

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
