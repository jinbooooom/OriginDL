#ifndef __ORIGIN_DL_MODEL_IO_H__
#define __ORIGIN_DL_MODEL_IO_H__

#include <string>
#include <unordered_map>
#include "../core/tensor.h"
#include "../nn/module.h"

namespace origin
{

// State Dict 类型定义（与 Module 中的定义一致）
using StateDict = std::unordered_map<std::string, Tensor>;

/**
 * @brief 保存 StateDict 到文件（.odl 格式）
 * @param state_dict 状态字典
 * @param filepath 文件路径（.odl 格式）
 * @note 类似 PyTorch 的 torch.save(model.state_dict(), "model.pth")
 */
void save(const StateDict &state_dict, const std::string &filepath);

/**
 * @brief 从文件加载对象（类似 PyTorch 的 torch.load）
 * @param filepath 文件路径
 * @return StateDict 对象（对于 .odl 文件）
 * @note 类似 PyTorch 的 torch.load("model.pth")
 *       根据文件扩展名自动判断格式：
 *       - .odl -> 返回 StateDict
 *       - .ckpt -> 返回 Checkpoint（未来支持）
 */
StateDict load(const std::string &filepath);

}  // namespace origin

#endif  // __ORIGIN_DL_MODEL_IO_H__
