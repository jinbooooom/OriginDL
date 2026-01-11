#include <getopt.h>
#include <cstdlib>
#include <iomanip>
#include <set>
#include "origin.h"

using namespace origin;
namespace F = origin::functional;
namespace nn = origin::nn;

/**
 * @brief 简单的CNN模型类
 * 结构：Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d -> Flatten -> Linear
 * -> ReLU -> Linear
 */
class SimpleCNN : public Module
{
private:
    // 第一层卷积：1通道 -> 64通道，3x3卷积核
    std::unique_ptr<nn::Conv2d> conv1_;

    // BatchNorm2d 批归一化层：64通道
    std::unique_ptr<nn::BatchNorm2d> bn1_;

    // ReLU 激活层
    std::unique_ptr<nn::ReLU> relu1_;

    // MaxPool2d 池化层
    std::unique_ptr<nn::MaxPool2d> maxpool1_;

    // 第二层卷积：64通道 -> 128通道，3x3卷积核
    std::unique_ptr<nn::Conv2d> conv2_;

    // BatchNorm2d 批归一化层：128通道
    std::unique_ptr<nn::BatchNorm2d> bn2_;

    // ReLU 激活层
    std::unique_ptr<nn::ReLU> relu2_;

    // MaxPool2d 池化层
    std::unique_ptr<nn::MaxPool2d> maxpool2_;

    // Flatten 层
    std::unique_ptr<nn::Flatten> flatten_;

    // 全连接层1：7*7*128 -> 128
    std::unique_ptr<nn::Linear> fc1_;

    // ReLU 激活层
    std::unique_ptr<nn::ReLU> relu3_;

    // 全连接层2：128 -> 10
    std::unique_ptr<nn::Linear> fc2_;

public:
    SimpleCNN()
        : conv1_(std::make_unique<nn::Conv2d>(1,
                                              64,
                                              std::make_pair(3, 3),
                                              std::make_pair(1, 1),
                                              std::make_pair(1, 1),
                                              true)),
          bn1_(std::make_unique<nn::BatchNorm2d>(64)),
          relu1_(std::make_unique<nn::ReLU>()),
          maxpool1_(std::make_unique<nn::MaxPool2d>(std::make_pair(2, 2), std::make_pair(2, 2), std::make_pair(0, 0))),
          conv2_(std::make_unique<nn::Conv2d>(64,
                                              128,
                                              std::make_pair(3, 3),
                                              std::make_pair(1, 1),
                                              std::make_pair(1, 1),
                                              true)),
          bn2_(std::make_unique<nn::BatchNorm2d>(128)),
          relu2_(std::make_unique<nn::ReLU>()),
          maxpool2_(std::make_unique<nn::MaxPool2d>(std::make_pair(2, 2), std::make_pair(2, 2), std::make_pair(0, 0))),
          flatten_(std::make_unique<nn::Flatten>()),
          fc1_(std::make_unique<nn::Linear>(7 * 7 * 128, 128, true)),
          relu3_(std::make_unique<nn::ReLU>()),
          fc2_(std::make_unique<nn::Linear>(128, 10, true))
    {
        // Conv2d、BatchNorm2d 和 Linear 层的参数已经通过 register_parameter 注册了
        // 由于它们继承自 Layer，Layer 继承自 Module，参数会自动被 Module 收集
    }

    // 重写 named_parameters 方法，手动收集所有子模块的参数
    std::unordered_map<std::string, Parameter *> named_parameters(const std::string &prefix = "")
    {
        std::unordered_map<std::string, Parameter *> named_params;

        // 收集当前模块的参数（如果有）
        auto base_params = Module::named_parameters(prefix);
        named_params.insert(base_params.begin(), base_params.end());

        // 手动收集所有子模块的参数
        auto conv1_params = conv1_->named_parameters(prefix.empty() ? "conv1" : prefix + ".conv1");
        auto bn1_params   = bn1_->named_parameters(prefix.empty() ? "bn1" : prefix + ".bn1");
        auto conv2_params = conv2_->named_parameters(prefix.empty() ? "conv2" : prefix + ".conv2");
        auto bn2_params   = bn2_->named_parameters(prefix.empty() ? "bn2" : prefix + ".bn2");
        auto fc1_params   = fc1_->named_parameters(prefix.empty() ? "fc1" : prefix + ".fc1");
        auto fc2_params   = fc2_->named_parameters(prefix.empty() ? "fc2" : prefix + ".fc2");

        named_params.insert(conv1_params.begin(), conv1_params.end());
        named_params.insert(bn1_params.begin(), bn1_params.end());
        named_params.insert(conv2_params.begin(), conv2_params.end());
        named_params.insert(bn2_params.begin(), bn2_params.end());
        named_params.insert(fc1_params.begin(), fc1_params.end());
        named_params.insert(fc2_params.begin(), fc2_params.end());

        return named_params;
    }

    // const 版本
    std::unordered_map<std::string, const Parameter *> named_parameters(const std::string &prefix) const
    {
        std::unordered_map<std::string, const Parameter *> named_params;

        // 收集当前模块的参数（如果有）
        auto base_params = Module::named_parameters(prefix);
        named_params.insert(base_params.begin(), base_params.end());

        // 手动收集所有子模块的参数
        auto conv1_params = conv1_->named_parameters(prefix.empty() ? "conv1" : prefix + ".conv1");
        auto bn1_params   = bn1_->named_parameters(prefix.empty() ? "bn1" : prefix + ".bn1");
        auto conv2_params = conv2_->named_parameters(prefix.empty() ? "conv2" : prefix + ".conv2");
        auto bn2_params   = bn2_->named_parameters(prefix.empty() ? "bn2" : prefix + ".bn2");
        auto fc1_params   = fc1_->named_parameters(prefix.empty() ? "fc1" : prefix + ".fc1");
        auto fc2_params   = fc2_->named_parameters(prefix.empty() ? "fc2" : prefix + ".fc2");

        named_params.insert(conv1_params.begin(), conv1_params.end());
        named_params.insert(bn1_params.begin(), bn1_params.end());
        named_params.insert(conv2_params.begin(), conv2_params.end());
        named_params.insert(bn2_params.begin(), bn2_params.end());
        named_params.insert(fc1_params.begin(), fc1_params.end());
        named_params.insert(fc2_params.begin(), fc2_params.end());

        return named_params;
    }

    // 重写 state_dict 方法，使用重写的 named_parameters
    StateDict state_dict() const override
    {
        StateDict state_dict;
        auto named_params = named_parameters("");
        for (auto &[name, param] : named_params)
        {
            // 将 Parameter 转换为 Tensor（Parameter 继承自 Tensor，可以直接转换）
            state_dict[name] = static_cast<const Tensor &>(*param);
        }
        return state_dict;
    }

    // 重写 load_state_dict 方法，使用重写的 named_parameters
    void load_state_dict(const StateDict &state_dict, bool strict = true) override
    {
        auto named_params = named_parameters("");
        std::set<std::string> loaded_keys;

        // 加载参数
        for (auto &[name, param] : named_params)
        {
            auto it = state_dict.find(name);
            if (it != state_dict.end())
            {
                // 检查形状是否匹配
                if (param->shape() != it->second.shape())
                {
                    THROW_RUNTIME_ERROR("Shape mismatch for parameter '{}': expected {}, got {}", name,
                                        param->shape().to_string(), it->second.shape().to_string());
                }
                // 更新参数值
                *param = Parameter(it->second);
                loaded_keys.insert(name);
            }
            else if (strict)
            {
                THROW_RUNTIME_ERROR("Missing parameter '{}' in state_dict (strict mode)", name);
            }
        }

        // 检查是否有未使用的键
        if (strict)
        {
            for (const auto &[key, value] : state_dict)
            {
                if (loaded_keys.find(key) == loaded_keys.end())
                {
                    THROW_RUNTIME_ERROR("Unexpected parameter '{}' in state_dict (strict mode)", key);
                }
            }
        }
    }

    Tensor forward(const Tensor &input) override
    {
        // 输入形状: (N, 784) -> reshape为 (N, 1, 28, 28)
        auto x = F::reshape(input, Shape{input.shape()[0], 1, 28, 28});

        // 第一层：Conv2d(1, 64, 3x3, pad=1) -> BatchNorm2d(64) -> ReLU -> MaxPool2d(2x2)
        x = conv1_->forward(x);
        x = bn1_->forward(x);
        x = relu1_->forward(x);
        x = maxpool1_->forward(x);
        // 形状: (N, 64, 14, 14)

        // 第二层：Conv2d(64, 128, 3x3, pad=1) -> BatchNorm2d(128) -> ReLU -> MaxPool2d(2x2)
        x = conv2_->forward(x);
        x = bn2_->forward(x);
        x = relu2_->forward(x);
        x = maxpool2_->forward(x);
        // 形状: (N, 128, 7, 7)

        // Flatten: (N, 128, 7, 7) -> (N, 128*7*7) = (N, 6272)
        x = flatten_->forward(x);

        // 全连接层1：6272 -> 128
        x = fc1_->forward(x);
        x = relu3_->forward(x);
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
        auto bn1_params   = bn1_->parameters();
        auto conv2_params = conv2_->parameters();
        auto bn2_params   = bn2_->parameters();
        auto fc1_params   = fc1_->parameters();
        auto fc2_params   = fc2_->parameters();

        params.insert(params.end(), conv1_params.begin(), conv1_params.end());
        params.insert(params.end(), bn1_params.begin(), bn1_params.end());
        params.insert(params.end(), conv2_params.begin(), conv2_params.end());
        params.insert(params.end(), bn2_params.begin(), bn2_params.end());
        params.insert(params.end(), fc1_params.begin(), fc1_params.end());
        params.insert(params.end(), fc2_params.begin(), fc2_params.end());

        // 注意：ReLU、MaxPool2d、Flatten 层没有参数，不需要收集

        return params;
    }

    void to(Device device) override
    {
        // 首先迁移当前模块自己的参数（如果有的话）
        Module::to(device);

        // 迁移所有层到指定设备
        conv1_->to(device);
        bn1_->to(device);
        relu1_->to(device);
        maxpool1_->to(device);
        conv2_->to(device);
        bn2_->to(device);
        relu2_->to(device);
        maxpool2_->to(device);
        flatten_->to(device);
        fc1_->to(device);
        relu3_->to(device);
        fc2_->to(device);
    }
};

/**
 * @brief 训练配置结构体
 */
struct TrainingConfig
{
    int max_epoch           = 10;
    int batch_size          = 256;
    float learning_rate     = 0.0001f;  // 降低学习率：0.0005 -> 0.0001，学习率太大，训练到后期越训精度越低
    float weight_decay_rate = 1e-4f;
    int log_interval        = 50;
    std::string model_path  = "model/mnist_model.odl";
    int checkpoint_interval = 5;
    int random_seed         = 42;

    /**
     * @brief 获取 checkpoint 目录（从 model_path 的目录派生）
     * @return checkpoint 目录路径
     */
    std::string checkpoint_dir() const
    {
        // 从 model_path 提取目录，然后添加 "checkpoints" 子目录
        size_t last_slash = model_path.find_last_of('/');
        if (last_slash != std::string::npos)
        {
            return model_path.substr(0, last_slash + 1) + "checkpoints";
        }
        else
        {
            return "checkpoints";
        }
    }

    /**
     * @brief 打印配置信息
     */
    void print() const
    {
        logi("=== Training Configuration ===");
        logi("Max epochs: {}", max_epoch);
        logi("Batch size: {}", batch_size);
        logi("Learning rate: {}", learning_rate);
        logi("Weight decay: {}", weight_decay_rate);
        logi("Log interval: {}", log_interval);
        logi("Model path: {}", model_path);
        logi("Checkpoint dir: {}", checkpoint_dir());
        logi("Checkpoint interval: {} epochs", checkpoint_interval);
        logi("Random seed: {}", random_seed);
        logi("==============================");
    }
};

/**
 * @brief 打印使用说明
 * @param program_name 程序名称
 */
void usage(const char *program_name)
{
    loga("Usage: %s [OPTIONS]\n", program_name);
    loga("Options:\n");
    loga("  -e, --epochs EPOCHS          Maximum number of epochs (default: 10)\n");
    loga("  -b, --batch-size SIZE        Batch size (default: 256)\n");
    loga("  -l, --learning-rate LR       Learning rate (default: 0.0001)\n");
    loga("  -w, --weight-decay RATE      Weight decay rate (default: 1e-4)\n");
    loga("  -i, --log-interval INTERVAL  Log interval in batches (default: 50)\n");
    loga("  -m, --model-path PATH        Path to save model (default: model/mnist_model.odl)\n");
    loga("  -c, --checkpoint-interval N  Save checkpoint every N epochs (default: 5)\n");
    loga("  -s, --seed SEED              Random seed (default: 42)\n");
    loga("  -h, --help                   Show this help message\n");
}

/**
 * @brief 解析命令行参数
 * @param argc 参数数量
 * @param argv 参数数组
 * @return TrainingConfig 配置对象
 */
TrainingConfig parse_args(int argc, char *argv[])
{
    TrainingConfig config;

    // 定义长选项
    static struct option long_options[] = {{"epochs", required_argument, 0, 'e'},
                                           {"batch-size", required_argument, 0, 'b'},
                                           {"learning-rate", required_argument, 0, 'l'},
                                           {"weight-decay", required_argument, 0, 'w'},
                                           {"log-interval", required_argument, 0, 'i'},
                                           {"model-path", required_argument, 0, 'm'},
                                           {"checkpoint-interval", required_argument, 0, 'c'},
                                           {"seed", required_argument, 0, 's'},
                                           {"help", no_argument, 0, 'h'},
                                           {0, 0, 0, 0}};

    int option_index = 0;
    int c;

    while ((c = getopt_long(argc, argv, "e:b:l:w:i:m:c:s:h", long_options, &option_index)) != -1)
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
            case 'l':
                config.learning_rate = std::atof(optarg);
                if (config.learning_rate <= 0.0f)
                {
                    logw("Invalid learning_rate: {}. Using default: 0.0001", optarg);
                    config.learning_rate = 0.0001f;
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
            case 'h':
                usage(argv[0]);
                std::exit(0);
            case '?':
                // getopt_long 已经打印了错误信息
                logw("Use -h or --help for usage information");
                break;
            default:
                break;
        }
    }

    return config;
}

int main(int argc, char *argv[])
{
    // 解析命令行参数
    TrainingConfig config = parse_args(argc, argv);

    // 设置随机种子
    std::srand(config.random_seed);

    // 检测并选择设备（GPU优先，如果没有GPU则使用CPU）
    Device device(DeviceType::kCPU);
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

    logi("=== MNIST Handwritten Digit Recognition with CNN ===");
    logi("Device: {}", device.to_string());
    config.print();

    // 加载数据集
    logi("Loading MNIST dataset...");
    MNIST train_dataset("./data/mnist", true);  // 训练集
    MNIST test_dataset("./data/mnist", false);  // 测试集

    logi("Train dataset size: {}", train_dataset.size());
    logi("Test dataset size: {}", test_dataset.size());

    // 创建数据加载器
    DataLoader train_loader(train_dataset, config.batch_size, true);  // 训练时打乱
    DataLoader test_loader(test_dataset, config.batch_size, false);   // 测试时不打乱

    // 创建模型
    logi("Creating CNN model...");
    SimpleCNN model;
    model.to(device);  // 将模型移到指定设备
    logi("Model created with {} parameters", model.parameters().size());

    // 创建优化器
    Adam optimizer(model, config.learning_rate);

    // 注册权重衰减Hook
    WeightDecay weight_decay(config.weight_decay_rate);
    optimizer.register_hook(weight_decay.hook());

    // 训练循环
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

        train_loader.reset();
        // while (train_loader.has_next() && train_batches < 10)
        while (train_loader.has_next())  // 完整训练：训练整个 epoch
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
                    auto guard = no_grad();
                    auto acc   = accuracy(y, t);
                    acc_value  = acc.item<float>();
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
                    cuda::synchronize();
                }

                train_correct += static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                train_total += current_batch_size;
                train_loss += loss_value;
                train_batches++;
                // train_iter_count++;  // 快速测试模式计数（已注释）

                // 根据log_interval控制打印频率
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
        // 快速测试模式（已注释，恢复完整测试）
        // const int max_test_iters = 5;  // 快速测试：只测试5个批次

        {
            auto guard = no_grad();  // 测试时禁用梯度计算
            test_loader.reset();
            // int test_iter_count = 0;  // 快速测试模式计数（已注释）
            while (test_loader.has_next())  // 完整测试：测试整个测试集
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
                    auto y         = model(x);
                    auto loss      = F::softmax_cross_entropy(y, t_int32);
                    float loss_val = loss.item<float>();

                    // 使用accuracy函数计算准确率
                    auto acc               = accuracy(y, t_int32);
                    float acc_value        = acc.item<float>();
                    int current_batch_size = static_cast<int>(x.shape()[0]);

                    int batch_correct = static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                    test_correct += batch_correct;
                    test_total += current_batch_size;
                    test_loss += loss_val;
                    test_batches++;
                    // test_iter_count++;  // 快速测试模式计数（已注释）

                    // 根据log_interval控制打印频率
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

        // 输出epoch结果
        logi("========== Epoch {}/{} Summary ==========", epoch + 1, config.max_epoch);
        logi("  Train Loss: {:.4f}, Train Acc: {:.2f}%", avg_train_loss, train_acc);
        logi("  Test Loss:  {:.4f}, Test Acc:  {:.2f}%", avg_test_loss, test_acc);
        logi("===========================================");

        // 保存 Checkpoint（每 N 个 epoch 或最后一个 epoch）
        if ((epoch + 1) % config.checkpoint_interval == 0 || (epoch + 1) == config.max_epoch)
        {
            try
            {
                // 确保 checkpoint 目录存在（忽略返回值）
                std::string ckpt_dir = config.checkpoint_dir();
                (void)system(("mkdir -p " + ckpt_dir).c_str());

                std::string checkpoint_path = ckpt_dir + "/checkpoint_epoch_" + std::to_string(epoch + 1) + ".ckpt";

                Checkpoint checkpoint;
                checkpoint.model_state_dict             = model.state_dict();
                checkpoint.optimizer_state_dict["adam"] = optimizer.state_dict();
                checkpoint.epoch                        = epoch + 1;
                checkpoint.step                         = train_batches;  // 当前总步数
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

    // 保存训练好的模型
    logi("Saving model to {}...", config.model_path);
    try
    {
        // 确保 model 目录存在（忽略返回值）
        (void)system("mkdir -p model");

        model.eval();  // 设置为评估模式
        save(model.state_dict(), config.model_path);
        logi("Model saved successfully to {}", config.model_path);
    }
    catch (const std::exception &e)
    {
        logw("Failed to save model: {}", e.what());
        return 1;
    }

    // 重新加载模型并进行推理测试
    logi("===========================================");
    logi("Reloading model and running inference test...");
    logi("===========================================");
    try
    {
        // 创建新模型并加载保存的参数
        SimpleCNN loaded_model;
        loaded_model.to(device);
        loaded_model.load(config.model_path);
        loaded_model.eval();  // 设置为评估模式
        logi("Model loaded successfully from {}", config.model_path);

        // 使用加载的模型进行完整的测试集推理
        logi("Running inference on full test set with loaded model...");
        float inference_loss  = 0.0f;
        int inference_batches = 0;
        int inference_correct = 0;
        int inference_total   = 0;

        {
            auto guard = no_grad();  // 推理时禁用梯度计算
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

                    // 使用加载的模型进行前向传播
                    auto y         = loaded_model(x);
                    auto loss      = F::softmax_cross_entropy(y, t_int32);
                    float loss_val = loss.item<float>();

                    // 使用accuracy函数计算准确率
                    auto acc               = accuracy(y, t_int32);
                    float acc_value        = acc.item<float>();
                    int current_batch_size = static_cast<int>(x.shape()[0]);

                    int batch_correct = static_cast<int>(acc_value * static_cast<float>(current_batch_size));
                    inference_correct += batch_correct;
                    inference_total += current_batch_size;
                    inference_loss += loss_val;
                    inference_batches++;
                }
            }
        }

        float avg_inference_loss = inference_loss / inference_batches;
        float inference_acc      = 100.0f * inference_correct / inference_total;

        logi("===========================================");
        logi("Inference Results (using loaded model):");
        logi("  Test Loss:  {:.4f}", avg_inference_loss);
        logi("  Test Acc:   {:.2f}%", inference_acc);
        logi("  Test Batches: {}", inference_batches);
        logi("===========================================");
    }
    catch (const std::exception &e)
    {
        logw("Failed to load model or run inference: {}", e.what());
        return 1;
    }

    return 0;
}
