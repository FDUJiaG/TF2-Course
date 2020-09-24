# 04-07 Optimizers

机器学习界有一群炼丹师，他们每天的日常是拿来药材（数据），架起八卦炉（模型），点着六味真火（优化算法），就摇着蒲扇等着丹药出炉了，不过，同样的食材，同样的菜谱，如果火候不一样，这出来的口味可是千差万别，火小了夹生，火大了易糊，火不匀则半生半糊

机器学习也是一样，模型优化算法的选择直接关系到最终模型的性能，有时候效果不好，未必是特征的问题或者模型设计的问题，很可能就是优化算法的问题

## A Framework to Review Optimization Algorithms

深度学习优化算法大概经历了 **SGD** → **SGDM** → **NAG** → **Adagrad** → **Adadelta(RMSprop)** → **Adam** → **Nadam** 这样的发展历程，在这里，用一个框架来梳理所有的优化算法，做一个更加高屋建瓴的对比

首先定义

$$
\begin{aligned}
Parameters &:\; \omega \\
Objective\ Function &: f(\omega) \\
Learning\ Rate &: \alpha \\
Epoch &: t
\end{aligned}
$$

而后，开始进行迭代优化，在每个 Epoch

1. 计算目标函数关于当前参数的梯度

    $$
    g_t = \nabla f(\omega_t)
    $$

1. 根据历史梯度计算一阶动量和二阶动量

    $$
    m_t = \phi(g_1, g_2,\cdots,g_t),\quad V_t = \psi(g_1, g_2,\cdots,g_t)
    $$

1. 计算当前时刻的下降梯度

    $$
    \eta_t = \alpha \cdot m_t / \sqrt{V_t}
    $$

1. 根据下降梯度进行更新

    $$
    \omega_{t+1} = \omega_t - \eta_t
    $$

掌握了这个框架，就可以轻轻松松设计自己的优化算法，根据这个框架，步骤 3 和 4 对于各个算法都是一致的，主要的差别就体现在 1 和 2 上

### SGD

先来看 SGD，SGD 没有动量的概念，也就是说

$$
m_t = g_t,\quad V_t = I^2
$$

代入步骤 3，可以看到下降梯度就是最简单的

$$
\eta_t = \alpha \cdot g_t
$$

SGD 最大的缺点是下降速度慢，而且可能会在沟壑的两边持续震荡，停留在一个局部最优点

### SGD with Momentum


为了抑制 SGD 的震荡，SGDM 认为梯度下降过程可以加入惯性，下坡的时候，如果发现是陡坡，那就利用惯性跑的快一些，在 SGD 基础上引入了一阶动量

$$
m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t
$$

也就是说，$t$ 时刻的下降方向，不仅由当前点的梯度方向决定，而且由此前累积的下降方向决定，根据经验

$$
\beta_1 \approx 0.9
$$

这就意味着下降方向主要是此前累积的下降方向，并略微偏向当前时刻的下降方向，想象高速公路上汽车转弯，在高速向前的同时略微偏向，急转弯可是要出事的

### SGD with Nesterov Acceleration

SGD 还有一个问题是困在局部最优的沟壑里面震荡，想象一下走到一个盆地，四周都是略高的小山，觉得没有下坡的方向，那就只能待在这里了，可是如果爬上高地，就会发现外面的世界还很广阔，因此，不能停留在当前位置去观察未来的方向，而要向前一步，多看一步，看远一些

NAG 全称 Nesterov Accelerated Gradient，是在 SGD，SGD-M 的基础上的进一步改进，改进点在于步骤 1，已经知道在时刻 $t$ 的主要下降方向是由累积动量决定的，自己的梯度方向说了也不算，那与其看当前梯度方向，不如先看看如果跟着累积动量走了一步，那个时候再怎么走，因此，NAG 在步骤 1 中，不计算当前位置的梯度方向，而是计算如果按照累积动量走了一步，那个时候的下降方向

$$
g_t = \nabla f(w_t - \alpha \cdot m_{t-1} / \sqrt{V_{t-1}})
$$

然后用下一个点的梯度方向，与历史累积动量相结合，计算步骤 2 中当前时刻的累积动量

### AdaGrad

此前我们都没有用到二阶动量，二阶动量的出现，才意味着 **自适应学习率** 优化算法时代的到来，SGD 及其变种以同样的学习率更新每个参数，但深度神经网络往往包含大量的参数，这些参数并不是总会用得到（想想大规模的 embedding），对于经常更新的参数，我们已经积累了大量关于它的知识，不希望被单个样本影响太大，希望学习速率慢一些；对于偶尔更新的参数，我们了解的信息太少，希望能从每个偶然出现的样本身上多学一些，即学习速率大一些

怎么样去度量历史更新频率呢？那就是二阶动量 —— 该维度上，迄今为止所有梯度值的平方和

$$
V_t = \sum\limits_{r=1}^t g_r^2
$$

再回顾一下步骤 3 中的下降梯度

$$
\eta_t = \alpha \cdot m_t / \sqrt{V_t}
$$

可以看出，此时实质上的学习率

$$
\alpha \to \alpha / \sqrt{V_t}
$$

一般为了分母为正，会增加一个小的平滑项，此外，参数更新越频繁，二阶动量越大，学习率就越小

这一方法在稀疏数据场景下表现非常好，但也存在一些问题，由于

$$
t \uparrow,\quad \sqrt{V_t} \uparrow, \quad \eta \to 0
$$

可能会使得训练过程提前结束，即便后续还有数据也无法学到必要的知识

### AdaDelta / RMSProp

由于 AdaGrad 单调递减的学习率变化过于激进，考虑一个改变二阶动量计算方法的策略，不累积全部历史梯度，而只关注过去一段时间窗口的下降梯度。这也就是 AdaDelta 名称中 Delta 的来历，修改的思路很简单，指数移动平均值大约就是过去一段时间的平均值，因此用这一方法来计算二阶累积动量

$$
V_t = \beta_1 \cdot V_{t-1} + (1-\beta_1) \cdot g_t^2
$$

这就避免了二阶动量持续累积、导致训练过程提前结束的问题了

### Adam

谈到这里，Adam 和 Nadam 的出现就很自然而然了 —— 它们是前述方法的集大成者

- SGD-M 在 SGD 基础上增加了一阶动量，
- AdaGrad 和 AdaDelta 在 SGD 基础上增加了二阶动量

把一阶动量和二阶动量都用起来，就有

$$
\mathrm{Adam = Adaptive + Momentum}
$$

- SGD 的一阶动量

    $$
    m_t = \beta_1 \cdot m_{t-1} + (1-\beta_1) \cdot g_t
    $$

- 加上 AdaDelta 的二阶动量

    $$
    V_t = \beta_1 \cdot V_{t-1} + (1-\beta_1) \cdot g_t^2
    $$

优化算法里最常见的两个超参数 $\beta_1, \beta_2$ 就都在这里了，前者控制一阶动量，后者控制二阶动量

### Nadam

虽说 Adam 是集大成者，但它居然遗漏了 Nesterov，这还能忍？必须给它加上，按照 NAG 的步骤 1

$$
g_t = \nabla f(w_t - \alpha \cdot m_{t-1} / \sqrt{V_{t-1}})
$$

就有所谓的

$$
\mathrm{Nesterov + Adam = Nadam}
$$


说到这里，大概可以理解为什么经常有人说 Adam / Nadam 目前最主流、最好用的优化算法了，对于一般新手炼丹师，优化器直接使用Adam，并使用其默认参数就足够了

一些爱写论文的炼丹师由于追求评估指标效果，可能会偏爱前期使用 Adam 优化器快速下降，后期使用 SGD 并精调优化器参数得到更好的结果

此外目前也有一些前沿的优化算法，据称效果比 Adam 更好，例如 LazyAdam，Look-ahead，RAdam，Ranger 等

## Use of Optimizers

优化器主要使用 `apply_gradients` 方法传入变量和对应梯度从而来对给定变量进行迭代，或者直接使用 `minimize` 方法对目标函数进行迭代优化

当然，更常见的使用是在编译时将优化器传入 Keras 的 Model，通过调用 `model.fit` 实现对 Loss 的的迭代优化

初始化优化器时会创建一个变量 `optimier.iterations` 用于记录迭代的次数，因此优化器和 `tf.Variable` 一样，一般需要在 `@tf.function` 外创建

首先再次构建打印时间函数

```python
import tensorflow as tf
import numpy as np 

# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m)) == 1:
            return(tf.strings.format("0{}", m))
        else:
            return(tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite), 
                timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)
```

以求解函数

$$
f(x) = a\cdot x^2 + b\cdot x + c
$$

的最小值为例

### Use optimizer.apply_gradients

```python
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    while tf.constant(True):
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
        
        # 迭代终止条件
        if tf.abs(dy_dx) < tf.constant(0.00001):
            break
            
        if tf.math.mod(optimizer.iterations, 100) == 0:
            printbar()
            tf.print("[Step " + tf.as_string(optimizer.iterations) + "]", end=" ")
            tf.print("x =", x)
                
    y = a * tf.pow(x, 2) + b * x + c
    return y

tf.print("\nmin(y) =", minimizef(), end=", ")
tf.print("x =", x)
```

**output**

```console
================================================================================14:10:27
[Step 100] x = 0.867380381
================================================================================14:10:27
[Step 200] x = 0.98241204
================================================================================14:10:27
[Step 300] x = 0.997667611
================================================================================14:10:27
[Step 400] x = 0.999690711
================================================================================14:10:27
[Step 500] x = 0.999959
================================================================================14:10:27
[Step 600] x = 0.999994516

min(y) = 0, x = 0.999995232
```

### Use optimizer.minimize

```python
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return(y)

@tf.function
def train(epoch=1000):
    for _ in tf.range(epoch):
        optimizer.minimize(f, [x])
    tf.print("epoch =", optimizer.iterations)
    return(f())

train(1000)
tf.print("min(y) =", f(), end=", ")
tf.print("x =", x)
```

**output**

```console
epoch = 1000
min(y) = 0, x = 0.999998569
```

### Use model.fit

```python
tf.keras.backend.clear_session()

class FakeModel(tf.keras.models.Model):
    def __init__(self, a, b, c):
        super(FakeModel, self).__init__()
        self.a = a
        self.b = b
        self.c = c
    
    def build(self):
        self.x = tf.Variable(0.0, name="x")
        self.built = True
    
    def call(self, features):
        loss  = self.a * (self.x) ** 2 + self.b * (self.x) + self.c
        return(tf.ones_like(features) * loss)
    
def myloss(y_true, y_pred):
    return tf.reduce_mean(y_pred)

model = FakeModel(tf.constant(1.0), tf.constant(-2.0), tf.constant(1.0))

model.build()
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
    loss = myloss
)
history = model.fit(
    tf.zeros((100, 2)),
    tf.ones(100), batch_size=1, epochs=10
)  # 迭代 1000 次
```

**output**

```console
Model: "fake_model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Total params: 1
Trainable params: 1
Non-trainable params: 0
_________________________________________________________________
Epoch 1/10
100/100 [==============================] - 0s 667us/step - loss: 0.2481
Epoch 2/10
100/100 [==============================] - 0s 784us/step - loss: 0.0044
Epoch 3/10
100/100 [==============================] - 0s 650us/step - loss: 7.6740e-05
Epoch 4/10
100/100 [==============================] - 0s 618us/step - loss: 1.3500e-06
Epoch 5/10
100/100 [==============================] - 0s 645us/step - loss: 1.8477e-08
Epoch 6/10
100/100 [==============================] - 0s 630us/step - loss: 0.0000e+00
Epoch 7/10
100/100 [==============================] - 0s 634us/step - loss: 0.0000e+00
Epoch 8/10
100/100 [==============================] - 0s 611us/step - loss: 0.0000e+00
Epoch 9/10
100/100 [==============================] - 0s 616us/step - loss: 0.0000e+00
Epoch 10/10
100/100 [==============================] - 0s 755us/step - loss: 0.0000e+00
```

查看结果

```python
tf.print("x =", model.x)
tf.print("loss =", model(tf.constant(0.0)))
```

**output**

```console
x = 0.999998569
loss = 0
```

## Built-in Optimizer

正如前面所说，深度学习优化算法大概经历了 **SGD** → **SGDM** → **NAG** → **Adagrad** → **Adadelta(RMSprop)** → **Adam** → **Nadam** 这样的发展历程，在 `keras.optimizers` 子模块中，它们基本上都有对应的类的实现

-  **SGD**，默认参数为纯 SGD，设置 `momentum` 参数不为 `0` 实际上变成 **SGDM**，考虑了一阶动量，设置 `nesterov` 为 `True` 后变成 **NAG**，即 Nesterov Accelerated Gradient，在计算梯度时计算的是向前走一步所在位置的梯度
- **Adagrad**，考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，缺点是学习率单调下降，可能后期学习速率过慢乃至提前停止学习
- **RMSprop**，考虑了二阶动量，对于不同的参数有不同的学习率，即自适应学习率，对 Adagrad 进行了优化，通过指数平滑只考虑一定窗口内的二阶动量
- **Adadelta**，考虑了二阶动量，与 RMSprop 类似，但是更加复杂一些，自适应性更强
- **Adam**，同时考虑了一阶动量和二阶动量，可以看成 RMSprop 上进一步考虑了一阶动量
- **Nadam**，在 Adam 基础上进一步考虑了 Nesterov Acceleration
