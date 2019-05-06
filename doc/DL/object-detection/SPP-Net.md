
## [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
### SPP-Net要解决的问题是什么？
spp提出的初衷是为了解决CNN对输入图片尺寸的限制。由于全连接层的存在，与之相连的最后一个卷积层的输出特征需要固定尺寸，从而要求输入图片尺寸也要固定。SPP-Net之前的做法是将图片裁剪或缩放（crop/warp），如下图所示:
![spp-net](sources/spp-net1.png)  
但裁剪或者缩放，会损失图片信息，因此提出了Spatial Pyramid Pooling,可以将任意size的feature map池化成固定大小。  
SPP-Net对这些网络中存在的缺点进行了改进，基本思想是，输入整张图像，提取出整张图像的特征图，然后利用空间关系从整张图像的特征图中，在spatial pyramid pooling layer提取各个region proposal的特征。  

### 介绍下空间金字塔池化
空间金字塔池化（spatial pyramid pooling）的网络结构如下图：
![spp](sources/spp-net2.png)  
将任意尺寸的feature map用三个尺度的金字塔层分别池化，将池化后的结果拼接得到固定长度的特征向量（图中的256为filter的个数），送入全连接层进行后续操作。  
（后来的Fast RCNN网络即借鉴了spp的思想。其中的ROI Pooling可理解为单尺度的SPP层）
（将4\*4,2\*2,1\*1网格放到feature map上，可以得到1+4+16=21个不同的块(spatial bin)，从21个块中，每一个块使用max pooling，可得到21个特征向量）      
![spp-net3](sources/spp-net3.jpg)

#### 推荐/参考链接
- [Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/abs/1406.4729)
- [SppNet中的spatial pyramid pooling](https://www.jianshu.com/p/884c2828cd8e)
- [SPP-Net](https://www.jianshu.com/p/b2fa1df5e982)
- [【目标检测】SPP-net](https://zhuanlan.zhihu.com/p/60919662)
