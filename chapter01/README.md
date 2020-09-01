# TensorFlow Modeling Process


尽管 TensorFlow 设计上足够灵活，可以用于进行各种复杂的数值计算，但通常人们使用 TensorFlow 来实现机器学习模型，尤其常用于实现神经网络模型

从原理上说可以使用张量构建计算图来定义神经网络，并通过自动微分机制训练模型，但为简洁起见，一般 **推荐使用** TensorFlow 的高层次 Keras 接口来实现神经网络网模型

## General Process of Neural Network Modeling

使用 TensorFlow 实现神经网络模型的一般流程包括

1. 准备数据
1. 定义模型
1. 训练模型
1. 评估模型
1. 使用模型
1. 保存模型


**对新手来说，其中最困难的部分实际上是准备数据过程**

## Data Types Commonly Used for Modeling

我们在实践中通常会遇到的数据类型包括

- 结构化数据
- 图片数据
- 文本数据
- 时间序列数据

我们将分别以 Titanic 生存预测问题，Cifar2 图片分类问题，Imdb 电影评论分类问题，国内新冠疫情结束时间预测问题为例，演示应用 TensorFlow 对这四类数据的建模方法