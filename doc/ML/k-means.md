
## k-means

算法思想：

```
选择K个点作为初始质心  
反复执行：  
    计算每个点到质心的距离，将每个点指派到最近的质心，形成K个簇  
    重新计算每个簇的新质心  
直到： 簇不发生变化或达到最大迭代次数  
```

这里的重新计算每个簇的质心，如何计算的是根据目标函数得来的，因此在开始时我们要考虑距离度量和目标函数。

考虑欧几里得距离的数据，使用误差平方和（Sum of the Squared Error,SSE）作为聚类的目标函数，两次运行K均值产生的两个不同的簇集，我们更喜欢SSE最小的那个。


### 推荐/参考链接
- [5 分钟带你弄懂 k-means 聚类](https://blog.csdn.net/huangfei711/article/details/78480078)
- [深入理解K-Means聚类算法](https://blog.csdn.net/taoyanqi8932/article/details/53727841)

