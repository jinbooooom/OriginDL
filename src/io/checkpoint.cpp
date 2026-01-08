#include "origin/io/checkpoint.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include "origin/io/model_io.h"
#include "origin/utils/exception.h"

namespace origin
{

namespace fs = std::filesystem;

void save(const Checkpoint &checkpoint, const std::string &filepath)
{
    // .ckpt 格式是目录结构
    std::string ckpt_dir = filepath;
    // 如果文件路径以 .ckpt 结尾，去掉扩展名作为目录名
    if (ckpt_dir.size() > 5 && ckpt_dir.substr(ckpt_dir.size() - 5) == ".ckpt")
    {
        ckpt_dir = ckpt_dir.substr(0, ckpt_dir.size() - 5);
    }

    // 创建目录
    try
    {
        fs::create_directories(ckpt_dir);
    }
    catch (const std::exception &e)
    {
        THROW_RUNTIME_ERROR("Failed to create checkpoint directory '{}': {}", ckpt_dir, e.what());
    }

    // 保存模型参数（复用 .odl 格式）
    std::string model_path = ckpt_dir + "/model.odl";
    save(checkpoint.model_state_dict, model_path);

    // 保存优化器状态（复用 .odl 格式，但需要特殊处理）
    // 注意：优化器状态是 std::any 类型，不能直接保存为 .odl
    // 这里我们暂时只保存优化器配置，不保存缓冲区
    // 未来可以扩展为支持优化器状态的完整保存
    std::string optimizer_path = ckpt_dir + "/optimizer.odl";
    // 暂时跳过优化器状态的保存，因为 StateDict 不支持 std::any

    // 保存元数据（JSON 格式）
    std::string metadata_path = ckpt_dir + "/metadata.json";
    std::ofstream metadata_file(metadata_path);
    if (!metadata_file.is_open())
    {
        THROW_RUNTIME_ERROR("Failed to open metadata file for writing: {}", metadata_path);
    }

    metadata_file << "{\n";
    metadata_file << "  \"format_version\": \"1.0\",\n";
    metadata_file << "  \"format_type\": \"origindl_checkpoint\",\n";
    metadata_file << "  \"epoch\": " << checkpoint.epoch << ",\n";
    metadata_file << "  \"step\": " << checkpoint.step << ",\n";
    metadata_file << "  \"loss\": " << checkpoint.loss << ",\n";
    metadata_file << "  \"optimizer_type\": \"" << checkpoint.optimizer_type << "\",\n";
    metadata_file << "  \"optimizer_config\": {\n";
    bool first = true;
    for (const auto &[key, value] : checkpoint.optimizer_config)
    {
        if (!first)
        {
            metadata_file << ",\n";
        }
        metadata_file << "    \"" << key << "\": " << value;
        first = false;
    }
    metadata_file << "\n  }\n";
    metadata_file << "}\n";
    metadata_file.close();
}

Checkpoint load_checkpoint(const std::string &filepath)
{
    // .ckpt 格式是目录结构
    std::string ckpt_dir = filepath;
    // 如果文件路径以 .ckpt 结尾，去掉扩展名作为目录名
    if (ckpt_dir.size() > 5 && ckpt_dir.substr(ckpt_dir.size() - 5) == ".ckpt")
    {
        ckpt_dir = ckpt_dir.substr(0, ckpt_dir.size() - 5);
    }

    // 检查目录是否存在
    if (!fs::exists(ckpt_dir) || !fs::is_directory(ckpt_dir))
    {
        THROW_RUNTIME_ERROR("Checkpoint directory does not exist: {}", ckpt_dir);
    }

    Checkpoint checkpoint;

    // 加载模型参数
    std::string model_path = ckpt_dir + "/model.odl";
    if (fs::exists(model_path))
    {
        checkpoint.model_state_dict = load(model_path);
    }
    else
    {
        THROW_RUNTIME_ERROR("Model file not found in checkpoint: {}", model_path);
    }

    // 加载元数据
    std::string metadata_path = ckpt_dir + "/metadata.json";
    if (fs::exists(metadata_path))
    {
        std::ifstream metadata_file(metadata_path);
        if (!metadata_file.is_open())
        {
            THROW_RUNTIME_ERROR("Failed to open metadata file: {}", metadata_path);
        }

        // 简单的 JSON 解析（只解析基本字段）
        std::string line;
        while (std::getline(metadata_file, line))
        {
            // 解析 epoch
            if (line.find("\"epoch\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    // 移除空格和逗号
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.epoch = std::atoi(value_str.c_str());
                }
            }
            // 解析 step
            if (line.find("\"step\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.step = std::atoi(value_str.c_str());
                }
            }
            // 解析 loss
            if (line.find("\"loss\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.loss = std::atof(value_str.c_str());
                }
            }
            // 解析 optimizer_type
            if (line.find("\"optimizer_type\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    // 提取引号中的字符串
                    size_t quote_start = value_str.find('"');
                    size_t quote_end   = value_str.find('"', quote_start + 1);
                    if (quote_start != std::string::npos && quote_end != std::string::npos)
                    {
                        checkpoint.optimizer_type = value_str.substr(quote_start + 1, quote_end - quote_start - 1);
                    }
                }
            }
            // 解析 optimizer_config（解析所有配置项）
            if (line.find("\"lr\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.optimizer_config["lr"] = std::atof(value_str.c_str());
                }
            }
            if (line.find("\"beta1\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.optimizer_config["beta1"] = std::atof(value_str.c_str());
                }
            }
            if (line.find("\"beta2\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.optimizer_config["beta2"] = std::atof(value_str.c_str());
                }
            }
            if (line.find("\"eps\"") != std::string::npos)
            {
                size_t colon_pos = line.find(':');
                if (colon_pos != std::string::npos)
                {
                    std::string value_str = line.substr(colon_pos + 1);
                    value_str.erase(0, value_str.find_first_not_of(" \t,"));
                    value_str.erase(value_str.find_last_not_of(" \t,") + 1);
                    checkpoint.optimizer_config["eps"] = std::atof(value_str.c_str());
                }
            }
        }
        metadata_file.close();
    }
    else
    {
        THROW_RUNTIME_ERROR("Metadata file not found in checkpoint: {}", metadata_path);
    }

    return checkpoint;
}

}  // namespace origin
