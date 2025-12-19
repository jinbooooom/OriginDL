# TODO List

- [x] 25/10/10：跑通所有的单元测试
- [x] 25/10/13：支持创建不同数据类型的张量（目前只支持float）
- [x] 增加自己写的后端矩阵计算库 OriginMat(CPU版框架)
- [x] 增加自己写的后端矩阵计算库 OriginMat(GPU版框架)
- [ ] Tensor 增加静态检查
- [x] 拆分 originMat，当前这个文件太长了
- [ ] OriginMat支持视图转置
- [x] 删除 arrayfire 相关代码
- [x] 异常打印行号
- [x] 统一cpu/cuda的单元测试
- [ ] 实现cuda算子 reshape, transpose, sum, pow, broadcast_to, sum_to, matmul, 
- [ ] 整理用户文档，与 pytorch 对齐
- [ ] libtorch C++ 中的 scalar 类型的变量，对pow运算可以存入指数，对 add 运算可以存入加数。可以借鉴
- [x] 支持0维度张量（标量），这样就统一了标量某些算子与标量的运算。
- [ ] MNIST demo 的核心功能已完成，可以正常运行。剩余主要是：架构层面的优化（循环引用根本解决方案）
         
