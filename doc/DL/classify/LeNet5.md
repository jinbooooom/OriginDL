### LeNet5
![LeNet5](../sources/LeNet5_1.png)    
![LeNet5](../sources/LeNet5.PNG)  
5 * 5 * 16 的 feature map 与 5 * 5 * 120 的 filter 用 valid 方式卷积。相当于输出一个 1 * 1 * 120 的 feature map，即线性向量。  
注 LeNet5 的卷积填充方式均为 valid，池化均为最大池化。