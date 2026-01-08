#ifndef __ORIGIN_DL_CHECKPOINT_H__
#define __ORIGIN_DL_CHECKPOINT_H__

#include <any>
#include <string>
#include <unordered_map>
#include "../io/model_io.h"

namespace origin
{

/**
 * @brief Checkpoint 结构体，包含完整的训练状态
 * @note 类似 PyTorch 的 checkpoint 字典
 */
struct Checkpoint
{
    StateDict model_state_dict;  // 模型参数
    std::unordered_map<std::string, std::unordered_map<std::string, std::any>>
        optimizer_state_dict;  // 优化器状态（键为优化器名称，值为状态字典）
    int epoch  = -1;           // 当前 epoch
    int step   = -1;           // 当前 step
    float loss = 0.0f;         // 当前损失

    // 优化器类型和配置
    std::string optimizer_type;                               // "Adam", "SGD" 等
    std::unordered_map<std::string, float> optimizer_config;  // 学习率等配置
};

/**
 * @brief 保存 Checkpoint（类似 PyTorch 的 torch.save(checkpoint, "file.ckpt")）
 * @param checkpoint Checkpoint 对象
 * @param filepath 文件路径（.ckpt 格式，实际保存为目录）
 */
void save(const Checkpoint &checkpoint, const std::string &filepath);

/**
 * @brief 加载 Checkpoint（类似 PyTorch 的 torch.load("file.ckpt")）
 * @param filepath 文件路径（.ckpt 格式，实际是目录）
 * @return Checkpoint 对象
 */
Checkpoint load_checkpoint(const std::string &filepath);

}  // namespace origin

#endif  // __ORIGIN_DL_CHECKPOINT_H__
