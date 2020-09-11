# Mathematical Operations of the Tensor

- 张量数学运算主要有
    - 标量运算
    - 向量运算
    - 矩阵运算
- 张量运算的广播机制

## Scalar Operation

标量运算符主要包括

- 加减乘除乘方
- 三角函数
- 指数
- 对数

等常见函数，逻辑比较运算符等都是标量运算符

标量运算符的特点是对张量实施逐元素运算，有些标量运算符对常用的数学运算符进行了重载，并且支持类似 Numpy 的广播特性

许多标量运算符都在 `tf.math` 模块下

```python
import tensorflow as tf
import numpy as np
```

- 加法

```python
a = tf.constant([[1.0, 2], [-3, 4.0]])
b = tf.constant([[5.0, 6], [7.0, 8.0]])
a + b   # 运算符重载
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 6.,  8.],
       [ 4., 12.]], dtype=float32)>
```

- 减法

```python
a - b
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ -4.,  -4.],
       [-10.,  -4.]], dtype=float32)>
```

- 乘法（点乘）

```python
a * b 
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[  5.,  12.],
       [-21.,  32.]], dtype=float32)>
```

- 除法（点除）

```python
a / b
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 0.2       ,  0.33333334],
       [-0.42857143,  0.5       ]], dtype=float32)>
```

- 乘方

```python
a ** 2
```

**output**

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 1.,  4.],
       [ 9., 16.]], dtype=float32)>
```

- 指数运算可处理开方问题

```python
a ** 0.5
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1.       , 1.4142135],
       [      nan, 2.       ]], dtype=float32)>
```

- 取余

```python
a % 3 # mod 的运算符重载，等价于 m=tf.math.mod(a, 3)
```

**output**

```console
<tf.Tensor: shape=(3,), dtype=int32, numpy=array([1, 2, 0], dtype=int32)>
```

- 求商（保证余数是非负的）

```python
a // 3  # 地板除法
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[ 0.,  0.],
       [-1.,  1.]], dtype=float32)>
```

- 布尔运算

```python
a >= 2
```

**output**

```
<tf.Tensor: shape=(2, 2), dtype=bool, numpy=
array([[False,  True],
       [False,  True]])>
```

- 逻辑与

```python
(a >= 2) & (a <= 3)
```

**output**

```
<tf.Tensor: shape=(2, 2), dtype=bool, numpy=
array([[False,  True],
       [False, False]])>
```

- 逻辑或

```python
(a >= 2) | (a <= 3)
```

**output**

```
<tf.Tensor: shape=(2, 2), dtype=bool, numpy=
array([[ True,  True],
       [ True,  True]])>
```

- 逻辑判断

```python
a == 5  # tf.equal(a, 5)
```

**output**

```
<tf.Tensor: shape=(3,), dtype=bool, numpy=array([False, False, False])>
```

- 开方运算

```python
tf.sqrt(a)
```

**output**

```
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[1.       , 1.4142135],
       [      nan, 2.       ]], dtype=float32)>
```

- 多项求和

```python
a = tf.constant([1.0, 8.0])
b = tf.constant([5.0, 6.0])
c = tf.constant([6.0, 7.0])
tf.add_n([a, b, c])
```

**output**

```console
<tf.Tensor: shape=(2,), dtype=float32, numpy=array([12., 21.], dtype=float32)>
```

- 求最大值

```python
tf.print(tf.maximum(a, b))
```

**output**

```console
[5 8]
```

- 求最小值

```python
tf.print(tf.minimum(a, b))
```

**output**

```console
[1 6]
```

- 取整

```python
x = tf.constant([2.6, -2.7])

tf.print(tf.math.round(x))  # 保留整数部分，四舍五入
tf.print(tf.math.floor(x))  # 保留整数部分，向下归整
tf.print(tf.math.ceil(x))   # 保留整数部分，向上归整
```

**output**

```console
[3 -3]
[2 -3]
[3 -2]
```

- 幅值裁剪

```python
x = tf.constant([0.9, -0.8, 100.0, -20.0, 0.7])
y = tf.clip_by_value(x, clip_value_min=-1, clip_value_max=1)
z = tf.clip_by_norm(x, clip_norm=3)
tf.print(y)
tf.print(z)
```

**output**

```
[0.9 -0.8 1 -1 0.7]
[0.0264732055 -0.0235317405 2.94146752 -0.588293493 0.0205902718]
```

这里的 `clip_by_norm` 裁剪操作，可以将 `x` 中的每个值的绝对值放缩到 `clip_norm` 之内，以应对梯度爆炸的情况，具体计算方法如下 

$$
z = \frac{clip\_norm \cdot x}{\Vert x \Vert_2}
= \frac{clip\_norm}{\sqrt {\sum\limits_{i=1} x_i^2}} \cdot
\left[ \begin{matrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{matrix}\right]
$$

## Vector Operation


向量运算符只在一个特定轴上运算，将一个向量映射到一个标量或者另外一个向量，许多向量运算符都以 `reduce` 开头

```python
a = tf.range(1, 10)
tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(a))
tf.print(tf.reduce_max(a))
tf.print(tf.reduce_min(a))
tf.print(tf.reduce_prod(a))
```

**output**

```console
45
5
9
1
362880
```

- 张量指定维度进行 reduce

```python
b = tf.reshape(a, (3, 3))
tf.print(tf.reduce_sum(b, axis=1, keepdims=True))
tf.print(tf.reduce_sum(b, axis=0, keepdims=True))
```

**output**

```console
[[6]
 [15]
 [24]]
[[12 15 18]]
```

- bool 类型的 reduce

```python
p = tf.constant([True, False, False])
q = tf.constant([False, False, True])
tf.print(tf.reduce_all(p))
tf.print(tf.reduce_any(q))
```

**output**

```console
0
1
```

- 利用 `tf.foldr` 实现 `tf.reduce_sum`

```python
s = tf.foldr(lambda a, b: a + b, tf.range(10)) 
tf.print(s)
```

**output**

```console
45
```

- cum 扫描累积

```python
a = tf.range(1, 10)
tf.print(tf.math.cumsum(a))
tf.print(tf.math.cumprod(a))
```

**output**

```console
[1 3 6 ... 28 36 45]
[1 2 6 ... 5040 40320 362880]
```

- arg 最大最小值索引

```python
a = tf.range(1, 10)
tf.print(tf.argmax(a))
tf.print(tf.argmin(a))
```

**output**

```console
8
0
```

- `tf.math.top_k` 可以用于对张量排序

```python
a = tf.constant([9, 1, 3, 7, 5, 4, 8])

values, indices = tf.math.top_k(a, 3, sorted=True)
tf.print(values)
tf.print(indices)
```

**output**

```console
[8 7 5]
[5 2 3]
```

利用 `tf.math.top_k` 可以在 TensorFlow 中实现 KNN 算法

## Matrix Arithmetic

矩阵至少是二维的，类似 `tf.constant([1,2,3])` 这样的不是矩阵，除了一些常用的运算外，大部分和矩阵有关的运算都在 `tf.linalg` 子包中

- 矩阵乘法

```python
a = tf.constant([[1, 2], [3, 4]])
b = tf.constant([[2, 0], [0, 2]])
a @ b  # 等价于 tf.matmul(a, b)
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[2, 4],
       [6, 8]], dtype=int32)>
```

- 矩阵转置

```python
a = tf.constant([[1, 2], [3, 4]])
tf.transpose(a)
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
array([[1, 3],
       [2, 4]], dtype=int32)>
```

- 矩阵逆，必须为 tf.float32 或 tf.double 类型

```python
a = tf.constant([[1.0, 2], [3, 4]], dtype=tf.float32)
tf.linalg.inv(a)
```

**output**

```console
<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
array([[-2.0000002 ,  1.0000001 ],
       [ 1.5000001 , -0.50000006]], dtype=float32)>
```

- 矩阵求迹

```python
a = tf.constant([[1.0, 2], [3, 4]], dtype=tf.float32)
tf.linalg.trace(a)
```

**output**

```console
<tf.Tensor: shape=(), dtype=float32, numpy=5.0>
```

- 矩阵求范数

```python
a = tf.constant([[1.0, 2], [3, 4]])
tf.linalg.norm(a)
```

**output**

```console
<tf.Tensor: shape=(), dtype=float32, numpy=5.477226>
```

- 矩阵行列式

```python
a = tf.constant([[1.0, 2], [3, 4]])
tf.linalg.det(a)
```

**output**

```console
<tf.Tensor: shape=(), dtype=float32, numpy=-2.0>
```

- 矩阵特征值

```python
a = tf.constant([[1.0, 2], [-5, 4]])
tf.linalg.eigvals(a)
```

**output**

```console
<tf.Tensor: shape=(2,), dtype=complex64, numpy=array([2.4999995+2.7838817j, 2.5      -2.783882j ], dtype=complex64)>
```

- 矩阵 QR 分解, 将一个方阵分解为一个正交矩阵 q 和上三角矩阵 r，QR 分解实际上是对矩阵 a 实施 Schmidt 正交化得到 q

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
q,r = tf.linalg.qr(a)
tf.print(q)
tf.print(r)
tf.print(q @ r)
```

**output**

```console
[[-0.316227794 -0.948683321]
 [-0.948683321 0.316227734]]
[[-3.1622777 -4.4271884]
 [0 -0.632455349]]
[[1.00000012 1.99999976]
 [3 4]]
```

- 矩阵 svd 分解

```python
# svd 分解可以将任意一个矩阵分解为一个正交矩阵 u，一个对角阵 s 和一个正交矩阵 v.t() 的乘积
# svd 常用于矩阵压缩和降维
a  = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
s, u, v = tf.linalg.svd(a)
tf.print(u, "\n")
tf.print(s, "\n")
tf.print(v, "\n")
tf.print(u @ tf.linalg.diag(s) @ tf.transpose(v))
```

**output**

```console
[[0.229847744 -0.88346082]
 [0.524744868 -0.240782902]
 [0.819642067 0.401896209]] 

[9.52551842 0.51429987] 

[[0.619629562 0.784894466]
 [0.784894466 -0.619629562]] 

[[1.00000119 2]
 [3.00000095 4.00000048]
 [5.00000143 6.00000095]]
```

注意，可通过 `tf.transpose(u) @ u` 和 `v @ tf.transpose(v)` 来查看单位正交矩阵的效果

利用 svd 分解可以在 TensorFlow 中实现主成分分析降维

## Broadcasting Mechanism

TensorFlow 的广播规则和 Numpy 是一样的

1. 如果张量的维度不同，将维度较小的张量进行扩展，直到两个张量的维度都一样
1. 如果两个张量在某个维度上的长度是相同的，或者其中一个张量在该维度上的长度为 1，那么我们就说这两个张量在该维度上是相容的
1. 如果两个张量在所有维度上都是相容的，它们就能使用广播
1. 广播之后，每个维度的长度将取两个张量在该维度长度的较大值
1. 在任何一个维度上，如果一个张量的长度为1，另一个张量长度大于1，那么在该维度上，就好像是对第一个张量进行了复制

- 向量的广播机制使得可以在不同维数的情况依旧执行运算

```python
a = tf.constant([1, 2, 3])
b = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
b + a  # 等价于 b + tf.broadcast_to(a, b.shape)
```

**output**

```console
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [2, 3, 4],
       [3, 4, 5]], dtype=int32)>
```

- `tf.broadcast_to` 以显式的方式按照广播机制扩展张量的维度

```python
tf.broadcast_to(a, b.shape)
```

**output**

```console
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[1, 2, 3],
       [1, 2, 3],
       [1, 2, 3]], dtype=int32)>
```

- 计算广播后计算结果的形状，静态形状，TensorShape 类型参数

```python
tf.broadcast_static_shape(a.shape, b.shape)
```

**output**

```console
TensorShape([3, 3])
```

- 计算广播后计算结果的形状，动态形状，Tensor 类型参数

```python
c = tf.constant([1, 2, 3])
d = tf.constant([[1], [2], [3]])
tf.broadcast_dynamic_shape(tf.shape(c), tf.shape(d))
```

**output**

```console
<tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 3], dtype=int32)>
```

- 广播效果

```python
c + d # 等价于 tf.broadcast_to(c, [3, 3]) + tf.broadcast_to(d, [3, 3])
```

**output**

```console
<tf.Tensor: shape=(3, 3), dtype=int32, numpy=
array([[2, 3, 4],
       [3, 4, 5],
       [4, 5, 6]], dtype=int32)>
```
