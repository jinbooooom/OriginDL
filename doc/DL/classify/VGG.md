
## [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

[VGG](https://arxiv.org/abs/1409.1556)是Oxford的**V**isual **G**eometry **G**roup的组提出的。该网络是在ILSVRC 2014上的相关工作，主要工作是证明了增加网络的深度能够在一定程度上影响网络最终的性能。VGG有两种结构，分别是VGG16和VGG19，两者并没有本质上的区别，只是网络深度不一样。

VGG16相比AlexNet的一个改进是采用**连续的几个3x3的卷积核代替AlexNet中的较大卷积核（11x11，7x7，5x5）**。对于给定的感受野（输入图片的局部大小），**采用堆积的小卷积核是优于采用大的卷积核，因为堆叠的3*3卷积增加了网络深度，提供了更多数量的激活函数，多层非线性层可以来保证学习更复杂的模式，而且代价还比较小（参数更少）**。

简单来说，在VGG中，使用了3个3x3卷积核来代替7x7卷积核，使用了2个3x3卷积核来代替5*5卷积核，这样做的主要目的是在保证具有相同感受野的条件下，提升了网络的深度，在一定程度上提升了神经网络的效果。

比如，3个步长为1的3x3卷积核的一层层叠加作用可看成一个大小为7的感受野（其实就表示3个3x3连续卷积相当于一个7x7卷积），那么3个3x3的卷积核参数总量为 3x(3x3xC1xC2) ，如果直接使用77卷积核，一个7x7卷积核参数总量为 7x7xC1xC2 ，这里 C1，C2分别指的是输入和输出通道数。很明显，27xC1xC2小于49xC1xC2，即**相比一个7x7卷积核3个3x3卷积核减少了参数**；而且3x3卷积核有利于更好地保持图像性质。

下面是VGG网络的结构（VGG16和VGG19都在）：
![VGG](../sources/vgg.jpg)
- VGG16包含了16个隐藏层（13个卷积层和3个全连接层），如上图中的D列所示
- VGG19包含了19个隐藏层（16个卷积层和3个全连接层），如上图中的E列所示
VGG网络的结构非常一致，从头到尾全部使用的是3x3的卷积和2x2的max pooling。
如果你想看到更加形象化的VGG网络，可以使用[经典卷积神经网络（CNN）结构可视化工具](https://mp.weixin.qq.com/s/gktWxh1p2rR2Jz-A7rs_UQ)来查看高清无码的[VGG网络](https://dgschwend.github.io/netscope/#/preset/vgg-16)。
### 手画的vgg16详细过程
![手画的vgg16详细过程](../sources/VGG16.jpg)

### VGG优点
- VGGNet的结构非常简洁，整个网络都使用了同样大小的卷积核尺寸（3x3）和最大池化尺寸（2x2）。
- 几个小滤波器（3x3）卷积层的组合比一个大滤波器（5x5或7x7）卷积层好：
- 验证了通过不断加深网络结构可以提升性能。

### VGG缺点
VGG耗费更多计算资源，并且使用了更多的参数（这里不是3x3卷积的锅），导致更多的内存占用（140M）。其中绝大多数的参数都是来自于第一个全连接层。VGG有3个全连接层。  
有的文章称：发现这些全连接层即使被去除，对于性能也没有什么影响，这样就显著降低了参数数量。

注：很多pretrained的方法就是使用VGG的model（主要是16和19），VGG相对其他的方法，参数空间很大，最终的model有500多m，AlexNet只有200m，GoogLeNet更少，所以train一个vgg模型通常要花费更长的时间，所幸有公开的pretrained model让我们很方便的使用。

### 关于感受野：
假设你一层一层地重叠了3个3x3的卷积层（层与层之间有非线性激活函数）。在这个排列下，第一个卷积层中的每个神经元都对输入数据体有一个3x3的视野。第二个卷积层中的每一个神经元都对输入数据体有一个5x5的视野，第三个卷积层有7x7的视野。

### 推荐/参考：
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [CS231n Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [深度网络VGG理解](https://blog.csdn.net/wcy12341189/article/details/56281618)
- [深度学习经典卷积神经网络之VGGNet](https://blog.csdn.net/marsjhao/article/details/72955935)
- [VGG16 结构可视化](https://dgschwend.github.io/netscope/#/preset/vgg-16)
