#ifndef __ORIGIN_DL_MNIST_H__
#define __ORIGIN_DL_MNIST_H__

#include "dataset.h"
#include <string>
#include <vector>
#include <memory>

namespace origin
{

/**
 * @brief MNIST 手写数字数据集
 * 
 * 支持下载和加载 MNIST 数据，支持训练/测试集切换
 */
class MNIST : public Dataset
{
private:
    std::vector<std::vector<float>> images_;  // 图像数据，每个图像是 28x28 = 784 个像素
    std::vector<int32_t> labels_;              // 标签数据，0-9
    bool train_;                                // 是否为训练集
    std::string root_;                          // 数据存储根目录

    /**
     * @brief 读取 MNIST 图像文件
     * @param filepath 文件路径
     * @return 是否读取成功
     */
    bool load_images(const std::string &filepath);

    /**
     * @brief 读取 MNIST 标签文件
     * @param filepath 文件路径
     * @return 是否读取成功
     */
    bool load_labels(const std::string &filepath);

    /**
     * @brief 读取大端序整数（MNIST 文件格式）
     */
    uint32_t read_uint32(std::ifstream &file);

public:
    /**
     * @brief 构造函数
     * @param root 数据存储根目录，默认为 "./data"
     * @param train 是否为训练集，默认为 true
     */
    MNIST(const std::string &root = "./data", bool train = true);

    /**
     * @brief 获取单个数据项
     * @param index 数据索引
     * @return 数据对 (image, label)
     *         - image: 形状为 (784,) 的张量，像素值归一化到 [0, 1]
     *         - label: 形状为 () 的标量张量，值为 0-9
     */
    std::pair<Tensor, Tensor> get_item(size_t index) override;

    /**
     * @brief 获取数据集大小
     * @return 数据集中的样本数量
     */
    size_t size() const override;
};

}  // namespace origin

#endif  // __ORIGIN_DL_MNIST_H__

