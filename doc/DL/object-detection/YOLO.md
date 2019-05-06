## YOLO系列

﻿### YOLOv1: [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

YOLOv1是one-stage detection的开山之作，YOLO创造性的将物体检测任务直接当作回归问题（regression problem）来处理，将区域建议和检测两个阶段合二为一。在全图的范围作预测，很少因背景而误检。  

### 简单介绍下YOLOv1的预测过程
- 1.对一个输入图像（448\*448），首先将图像划分成S * S 的网格。
- 2.对于每个网格，预测B个包围框（每个box包含5个预测量，x, y, w, h和confidence）以及C个类别概率，总共输出S \* S \* (B * 5 + C) 个 tensor
- 3.根据上一步可以预测出S \* S \* B个目标窗口，然后根据阈值去除可能性比较低的目标窗口，再由NMS去除冗余窗口即可。
在 VOC数据集上，S=7，B=2，C=20。  
YOLOv1使用了end-to-end的回归方法，没有region proposal步骤，直接回归便完成了位置和类别的预测。由于YOLO网格设置比较稀疏，且每个网格只预测2个边界框，其总体预测精度不高，略低于Fast RCNN。其对小物体的检测效果较差，尤其是对密集的小物体表现比较差。
![YOLOv1net](sources/YOLOv1net.PNG)

### 如何理解损失函数
![loss](sources/YOLOv1_loss.png)  

﻿## YOLOv2：[YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242)

### YOLOv2 有哪些创新点？

YOLOv1虽然检测速度快，但在定位方面不够准确，并且召回率较低。为了提升定位准确度，改善召回率，YOLOv2在YOLOv1的基础上提出了几种改进策略：  

- Batch Normalization
  CNN在训练过程中网络每层输入的分布一直在改变, 会使训练过程难度加大，但可以通过normalize每层的输入解决这个问题。新的YOLO网络在每一个卷积层后添加batch normalization，通过这一方法，mAP获得了2%的提升。batch normalization 也有助于规范化模型，可以在舍弃dropout优化后依然不会过拟合。

- High Resolution Classifier
  目前的目标检测方法中，基本上都会使用ImageNet预训练过的模型（classifier）来提取特征，在YOLOv1中，分类网络的输入是256\*256，而检测时的网络输入是448\*448，分类网络分辨率不够高，给检测带来困难。为此，新的YOLO网络把分辨率直接提升到了448 * 448，这也意味着原有的网络模型必须进行某种调整以适应新的分辨率输入。  
  对于YOLOv2，作者首先对分类网络（自定义的darknet）进行了fine tune，分辨率改成448 * 448，在ImageNet数据集上训练10轮（10 epochs），训练后的网络就可以适应高分辨率的输入了。然后，作者对检测网络部分（也就是后半部分）也进行fine tune。这样通过提升输入的分辨率，mAP获得了4%的提升。

- Convolutional With Anchor Boxes
  YOLOv1利用全连接层的数据完成边框的预测，导致丢失较多的空间信息，定位不准。YOLOv2借鉴了Faster R-CNN中的anchor思想。（回顾一下，anchor是RNP网络中的一个关键步骤，说的是在卷积特征图上进行滑窗操作，每一个中心可以预测9种不同大小的建议框。看到YOLOv2的这一借鉴，我只能说SSD的作者是有先见之明的）  

为了引入anchor boxes来预测bounding boxes，作者在网络中果断去掉了全连接层。剩下的具体怎么操作呢？首先，作者去掉了后面的一个池化层以确保输出的卷积特征图有更高的分辨率。然后，通过缩减网络，让图片输入分辨率为416 * 416，这一步的目的是为了让后面产生的卷积特征图宽高都为奇数，这样就可以产生一个center cell。作者观察到，大物体通常占据了图像的中间位置， 就可以只用中心的一个cell来预测这些物体的位置，否则就要用中间的4个cell来进行预测，这个技巧可稍稍提升效率。最后，YOLOv2使用了卷积层降采样（factor为32），使得输入卷积网络的416 * 416图片最终得到13 * 13的卷积特征图（416/32=13）。   

加入了anchor boxes后，可以预料到的结果是召回率上升，准确率下降。我们来计算一下，假设每个cell预测9个建议框，那么总共会预测13 * 13 * 9 = 1521个boxes，而之前的网络仅仅预测7 * 7 * 2 = 98个boxes。具体数据为：没有anchor boxes，模型recall为81%，mAP为69.5%；加入anchor boxes，模型recall为88%，mAP为69.2%。这样看来，准确率只有小幅度的下降，而召回率则提升了7%，说明可以通过进一步的工作来加强准确率，的确有改进空间。

- Dimension Clusters（维度聚类）
  作者在使用anchor的时候遇到了两个问题，第一个是anchor boxes的宽高维度往往是精选的先验框（hand-picked priors），虽说在训练过程中网络也会学习调整boxes的宽高维度，最终得到准确的bounding boxes。但是，如果一开始就选择了更好的、更有代表性的先验boxes维度，那么网络就更容易学到准确的预测位置。

和以前的精选boxes维度不同，作者使用了K-means聚类方法类训练bounding boxes，可以自动找到更好的boxes宽高维度。传统的K-means聚类方法使用的是欧氏距离函数，也就意味着较大的boxes会比较小的boxes产生更多的error，聚类结果可能会偏离。为此，作者采用的评判标准是IOU得分（也就是boxes之间的交集除以并集），这样的话，error就和box的尺度无关。

## [YOLOv3: An Incremental Improvement](https://pjreddie.com/media/files/papers/YOLOv3.pdf)

### YOLOv3  有哪些创新？

- 提出新的网络结构：DarkNet-53  
  YOLOv3在之前Darknet-19的基础上引入了残差块，并进一步加深了网络，改进后的网络有53个卷积层，取名为Darknet-53。
- 融合FPN，多尺度预测  
  YOLOv3借鉴了FPN的思想，从不同尺度提取特征。YOLOv3提取最后3层特征图，不仅在每个特征图上分别独立做预测，同时将小特征图上采样到与大的特征图相同大小，然后与大的特征图拼接做进一步预测。用维度聚类的思想聚类出9种尺度的anchor box，将9种尺度的anchor box均匀的分配给3种尺度的特征图。每一个尺度的特征图预测 S \* S \* [3 \* (4 + 1 + C)]，这个式子与 YOLOv1 很像，但 YOLOv1 只在最后一个 7  \* 7 的特征图上做预测，但 YOLOv3 在 13 \* 13、26 \* 26、52  \* 52 的特征图上作预测，每层特征图有三个 anchor box。
- 用逻辑回归替代softmax作为分类器

#### 推荐/参考链接

- [从YOLOv1-到YOLOv3](https://blog.csdn.net/guleileo/article/details/80581858)

- [YOLO损失函数解析——你真的读懂YOLO了嘛？](https://blog.csdn.net/WenDayeye/article/details/88807190)
- [YOLO 系列损失函数理解](https://www.cnblogs.com/WSX1994/p/11226012.html)

- [深度学习500问](https://github.com/scutan90/DeepLearning-500-questions/blob/master/ch08_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B/%E7%AC%AC%E5%85%AB%E7%AB%A0_%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B.md)
- [YOLOv2 论文笔记](https://blog.csdn.net/jesse_mx/article/details/53925356)
- [目标检测网络之 YOLOv3](https://www.cnblogs.com/makefile/p/YOLOv3.html)