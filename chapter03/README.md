# TensorFlow Low-Level API

TensorFlow 的 **低阶 API** 主要包括 **张量操作**，**计算图** 和 **自动微分**，如果把模型比作一个房子，那么低阶API就是【模型之砖】

在低阶 API 层次上，可以把 TensorFlow 当做一个增强版的 Numpy 来使用，并且 TensorFlow 提供的方法比 Numpy 更全面，运算速度更快，如果需要的话，还可以使用 GPU 进行加速

前面几章我们对低阶 API 已经有了一个整体的认识，本章我们将重点详细介绍张量操作和 Autograph 计算图

## Tensor Operations

张量的操作主要包括

- [张量 **结构操作**](./c03-01-structural-operations-of-the-tensor.md)
    - 张量创建
    - 索引切片
    - 维度变换
    - 合并分割

- [张量 **数学运算**](./c03-02-mathematical-operations-of-the-tensor.md)
    - 标量运算
    - 向量运算
    - 矩阵运算

- 张量运算的 **广播机制**

## About Autograph

Autograph 计算图将介绍

- [使用 Autograph 的规范建议](./c03-03-rules-of-using-the-autograph.md)
- [Autograph 的机制原理](./c03-04-mechanisms-of-the-autograph.md)
- [Autograph 和 `tf.Module`](./c03-05-autograph-and-tf.Module.md)
