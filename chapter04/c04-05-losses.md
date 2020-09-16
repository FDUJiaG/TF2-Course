# 04-05 Loss Function

一般来说，监督学习的目标函数由损失函数和正则化项组成，即

$$
\mathrm{Objective = Loss + Regularization}
$$

对于 Keras 模型，目标函数中的正则化项一般在各层中指定，例如使用 Dense 的 `kernel_regularizer` 和 `bias_regularizer` 等参数指定权重使用 `l1` 或者 `l2` 正则化项，此外还可以用 `kernel_constraint` 和 `bias_constraint` 等参数约束权重的取值范围，这也是一种正则化手段

损失函数在模型编译时候指定

- 对于 **回归模型**，通常使用的损失函数是均方损失函数 `mean_squared_error`
- 对于 **二分类模型**，通常使用的是二元交叉熵损失函数 `binary_crossentropy`
- 对于 **多分类模型**
    - 如果 Label 是 One-Hot 编码的，则使用类别交叉熵损失函数 `categorical_crossentropy`
    - 如果 Label 是类别序号编码的，则需要使用稀疏类别交叉熵损失函数 `sparse_categorical_crossentropy`

如果有需要，也可以自定义损失函数，自定义损失函数需要接收两个张量 `y_true, y_pred` 作为输入参数，并输出一个标量作为损失函数值

首先导入一些必要的 Package

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, losses, regularizers, constraints
```

## Loss Functions and Regularization Terms

```python
tf.keras.backend.clear_session()

model = models.Sequential()
model.add(
    layers.Dense(
        64, input_dim=64,
        kernel_regularizer=regularizers.l2(0.01), 
        activity_regularizer=regularizers.l1(0.01),
        kernel_constraint=constraints.MaxNorm(max_value=2, axis=0)
    )
) 
model.add(
    layers.Dense(
        10,
        kernel_regularizer=regularizers.l1_l2(0.01, 0.01),
        activation="sigmoid"
    )
)
model.compile(
    optimizer="rmsprop",
    loss="binary_crossentropy",
    metrics=["AUC"]
)
model.summary()
```

**output**

```console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 64)                4160
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
=================================================================
Total params: 4,810
Trainable params: 4,810
Non-trainable params: 0
_________________________________________________________________
```

## Built-In Loss Function

内置的损失函数一般有 **类的实现** 和 **函数的实现** 两种形式，`CategoricalCrossentropy` 和 `categorical_crossentropy` 都是类别交叉熵损失函数，前者是类的实现形式，后者是函数的实现形式

常用的一些内置损失函数说明如下

- `mean_squared_error`，**均方误差损失**，用于回归，简写为 MSE，类与函数实现形式分别为 `MeanSquaredError` 和 `MSE`
- `mean_absolute_error`，**平均绝对值误差损失**，用于回归，简写为 MAE，类与函数实现形式分别为 `MeanAbsoluteError` 和 `MAE`
- `mean_absolute_percentage_error`，**平均百分比误差损失**，用于回归，简写为 **MAPE**, 类与函数实现形式分别为 `MeanAbsolutePercentageError` 和 `MAPE`
- `Huber`，**Huber 损失**，只有类实现形式，用于回归，介于 MSE 和 MAE 之间，对异常值比较鲁棒，相对 MSE 有一定的优势
- `binary_crossentropy`，**二元交叉熵**，用于二分类，类实现形式为 `BinaryCrossentropy`
- `categorical_crossentropy`，**类别交叉熵**，用于多分类，要求 Label 为 One-Hot 编码，类实现形式为 `CategoricalCrossentropy`
- `sparse_categorical_crossentropy`，**稀疏类别交叉熵**，用于多分类，要求 Label 为序号编码形式，类实现形式为 `SparseCategoricalCrossentropy`
- `hinge`，**合页损失函数**，用于二分类，最著名的应用是作为支持向量机 SVM 的损失函数，类实现形式为 `Hinge`
- `kld`，**相对熵损失**，也叫 **KL 散度**，常用于最大期望算法 EM 的损失函数，两个概率分布差异的一种信息度量，类与函数实现形式分别为 `KLDivergence` 或 `KLD`
- `cosine_similarity`，余弦相似度，可用于多分类，类实现形式为 `CosineSimilarity`

## Custom Loss Functions

自定义损失函数需要接收两个张量 `y_true, y_pred` 作为输入参数，并输出一个标量作为损失函数值，也可以对 `tf.keras.losses.Loss` 进行子类化，重写 `call` 方法实现损失的计算逻辑，从而得到损失函数的类的实现

下面是一个 Focal Loss 的自定义实现示范，Focal Loss 是一种对 `binary_crossentropy` 改进的损失函数形式
，它在样本不均衡和存在较多易分类的样本时相比 `binary_crossentropy` 具有明显的优势，它有两个可调参数

- `alpha` 参数，主要用于衰减负样本的权重
- `gamma` 参数，主要用于衰减容易训练样本的权重

从而让模型更加聚焦在正样本和困难样本上，这就是为什么这个损失函数叫做 Focal Loss

详见 [《5 分钟理解 Focal Loss 与 GHM —— 解决样本不平衡利器》](https://zhuanlan.zhihu.com/p/80594704)，核心函数为

$$
focal\_loss(y,\ p) =
\begin{cases}
    -\alpha  (1-p)^{\gamma}\log(p) & \text{if \quad y = 1}\\
    -(1-\alpha) p^{\gamma}\log(1-p) & \text{if \quad y = 0}
\end{cases}
$$

### Function Implementation

```python
def focal_loss(gamma=2., alpha=0.75):
    
    def focal_loss_fixed(y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce, axis=-1)
        return loss

    return focal_loss_fixed
```

### Class Implementation

```python
class FocalLoss(tf.keras.losses.Loss):
    
    def __init__(self, gamma=2.0, alpha=0.75, name="focal_loss"):
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        loss = tf.reduce_sum(alpha_factor * modulating_factor * bce, axis=-1)
        return loss
```

