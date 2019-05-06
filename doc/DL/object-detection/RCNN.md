## R-CNN系列
### 简单介绍下 R-CNN
- R-CNN提取 proposals并计算CNN 特征。利用选择性搜索（Selective Search）算法提取所有proposals（大约2000幅images），调整（resize/warp）它们成固定大小，以满足 CNN 输入要求（因为全连接层的限制），然后将 feature map 保存到本地磁盘。
- 训练 SVM。利用 feature map 训练SVM来对目标和背景进行分类（二分类的SVM，区分背景和目标）
- 边界框回归（Bounding boxes Regression）。用 Bounding Box 回归校正原来的 region proposal，生成预测窗口的坐标偏移值。

### Fast R-CNN 相比于 R-CNN 有什么创新点？
Fast R-CNN 是基于 R-CNN 和 SPP-Net 进行的改进。SPP-Net，其创新点在于计算整幅图像的 shared feature map，然后根据 object proposal 在 shared feature map 上映射到对应的 feature vector（就是不用重复计算feature map了）。当然，SPP-Net 也有缺点：和 R-CNN 一样，训练是多阶段（multiple-stage pipeline）的，速度还是不够"快"，特征还要保存到本地磁盘中。
- 1.只对整幅图像进行一次特征提取，避免 R-CNN 中的冗余特征提取
- 2.用 RoI pooling 层替换最后一层的 max pooling 层，同时引入建议框数据，提取相应建议框特征
- 3.Fast R-CNN 网络末尾采用并行的不同的全连接层，可同时输出分类结果和窗口回归结果，实现了 end-to-end 的多任务训练（建议框提取除外），也不需要额外的特征存储空间（R-CNN 中的特征需要保持到本地，来供SVM 和 Bounding-box regression 进行训练）
- 4.采用 SVD 对 Fast R-CNN 网络末尾并行的全连接层进行分解，减少计算复杂度，加快检测速度。

### ROI Pooling 是什么？
因为 Fast R-CNN 使用全连接层，所以应用 RoI Pooling 将不同大小的 ROI 转换为固定大小。  
RoI Pooling 是 Pooling 层的一种，而且是针对 RoI 的 Pooling，其特点是输入特征图尺寸不固定，但是输出特征图尺寸固定（如7x7）。

### Faster R-CNN有哪些创新点？
Fast R-CNN依赖于外部候选区域方法，如选择性搜索。但这些算法在CPU上运行且速度很慢。在测试中，Fast R-CNN需要2.3秒来进行预测，其中2秒用于生成2000个ROI。Faster R-CNN采用与Fast R-CNN相同的设计，只是它用内部深层网络代替了候选区域方法。新的候选区域网络（RPN）在生成ROI时效率更高，并且以每幅图像10毫秒的速度运行。  
![Faster R-CNN](sources/faster.png)

### 曾今做的Faster R-CNN笔记自己都看不懂了，有充足时间时再梳理，整理Faster-RCNN不是当前最迫切的事。第一个参考链接很nice。

#### 推荐/参考链接
- [Object Detection and Classification using R-CNNs](http://www.telesens.co/2018/03/11/object-detection-and-classification-using-r-cnns/)
- [RCNN系列总结（RCNN,SPPNET,Fast RCNN,Faster RCNN）](https://blog.csdn.net/hust_lmj/article/details/78974348)
- [【RCNN系列】【超详细解析】](https://blog.csdn.net/amor_tila/article/details/78809791)
- [From R-CNN to Faster R-CNN: The Evolution of Object Detection Technology](https://dzone.com/articles/from-r-cnn-to-faster-r-cnn-the-evolution-of-object)
- [目标检测技术演化：从R-CNN到Faster R-CNN](https://zhuanlan.zhihu.com/p/40679183)
- [一文读懂Faster R-CNN](https://zhuanlan.zhihu.com/p/31426458)

