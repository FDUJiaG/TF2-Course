# Data Structure of Tensor

如果我们将普通程序看成如下的等式

$$
程序 = 数据结构 + 算法
$$

那么，对于 TensorFlow 程序，上述等式可以推广为

$$
\bold{TensorFlow}\;程序 = 张量数据结构 + 计算图算法语言
$$

即张量和计算图是 TensorFlow 的核心概念

Tensorflow 的基本数据结构是张量 Tensor，张量即多维数组（或者说矩阵的概念），和 Numpy 中的 ndarray 很类似

从行为特性来看，有两种类型的张量

- 常量 Constant，其值在计算图中不可以被重新赋值
- 变量 Variable，可以在计算图中用 `assign` 等算子重新赋值

## Constant Tensor

张量的数据类型和 `numpy.array` 基本一一对应

```python
import numpy as np
import tensorflow as tf

i = tf.constant(1)                      # tf.int32 类型常量
l = tf.constant(1, dtype=tf.int64)      # tf.int64 类型常量
f = tf.constant(1.23)                   # tf.float32 类型常量
d = tf.constant(3.14, dtype=tf.double)  # tf.double 类型常量
s = tf.constant("hello world")          # tf.string 类型常量
b = tf.constant(True)                   # tf.bool 类型常量

print(tf.int64 == np.int64) 
print(tf.bool == np.bool)
print(tf.double == np.float64)
print(tf.string == np.unicode)          # tf.string 类型和 np.unicode 类型不等价
```

**output**

```console
True
True
True
False
```

不同类型的数据可以用不同维度（rank）的张量来表示，可以简单地总结为，有几层中括号，就是多少维的张量

### Scalar

标量为 0 维张量

```python
scalar = tf.constant(True)  # 标量，0 维张量

print(tf.rank(scalar))
print(scalar.numpy().ndim)  # tf.rank 的作用和 numpy 的 ndim 方法相同
```

**output**

```console
tf.Tensor(0, shape=(), dtype=int32)
0
```

### Vectors

向量为 1 维张量

```python
vector = tf.constant([1.0, 2.0, 3.0, 4.0])  # 向量，1 维张量

print(tf.rank(vector))
print(np.ndim(vector.numpy()))
```

**output**

```console
tf.Tensor(1, shape=(), dtype=int32)
1
```

### Matrix

矩阵为 2 维张量

```python
matrix = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # 矩阵, 2 维张量

print(tf.rank(matrix).numpy())
print(np.ndim(matrix))
```

**output**

```console
2
2
```

### 3D Tensor

彩色图像有 rgb 三个通道，可以表示为 3 维张量

```python
tensor3 = tf.constant([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]) # 3 维张量
print(tensor3)
print(tf.rank(tensor3))
```

**output**

```console
tf.Tensor(
[[[1. 2.]
  [3. 4.]]

 [[5. 6.]
  [7. 8.]]], shape=(2, 2, 2), dtype=float32)
tf.Tensor(3, shape=(), dtype=int32)
```

### 4D Tensor

视频还有时间维，可以表示为 4 维张量

```python
tensor4 = tf.constant([[[[1.0, 1.0], [2.0, 2.0]], [[3.0, 3.0], [4.0, 4.0]]],
                        [[[5.0, 5.0], [6.0, 6.0]], [[7.0, 7.0], [8.0, 8.0]]]])  # 4 维张量
print(tensor4)
print(tf.rank(tensor4))
```

**output**

```console
tf.Tensor(
[[[[1. 1.]
   [2. 2.]]

  [[3. 3.]
   [4. 4.]]]


 [[[5. 5.]
   [6. 6.]]

  [[7. 7.]
   [8. 8.]]]], shape=(2, 2, 2, 2), dtype=float32)
tf.Tensor(4, shape=(), dtype=int32)
```

### Change Data Type

可以用 `tf.cast` 改变张量的数据类型

```python
h = tf.constant([123, 456], dtype=tf.int32)
f = tf.cast(h,tf.float32)
print(h.dtype, f.dtype)
```

**output**

```console
<dtype: 'int32'> <dtype: 'float32'>
```

### Change to Numpy

可以用 `numpy()` 方法将 TensorFlow 中的张量转化成 Numpy 中的张量，并使用 `shape` 方法查看张量的尺寸

```python
y = tf.constant([[1.0, 2.0], [3.0, 4.0]])
print(y.numpy())    # 转换成 np.array
print(y.shape)
```

**output**

```console
[[1. 2.]
 [3. 4.]]
(2, 2)
```

### Chinese Transcoding

可以用 `shape` 方法查看张量的尺寸

```python
u = tf.constant(u"你好 世界")
print(u.numpy())  
print(u.numpy().decode("utf-8"))
```

**output**

```console
b'\xe4\xbd\xa0\xe5\xa5\xbd \xe4\xb8\x96\xe7\x95\x8c'
你好 世界
```

## Variable Tensor

模型中需要被训练的参数一般被设置成变量

### Constant Values cannot be Changed

```python
# 常量值不可以改变，常量的重新赋值相当于创造新的内存空间
c = tf.constant([1.0, 2.0])
print(c)
print(id(c))
c = c + tf.constant([1.0, 1.0])
print(c)
print(id(c))
```

**output**

```console
tf.Tensor([1. 2.], shape=(2,), dtype=float32)
4451053232
tf.Tensor([2. 3.], shape=(2,), dtype=float32)
4450253280
```

### Variable Values can be Changed

```python
# 变量的值可以改变，可以通过 assign, assign_add 等方法给变量重新赋值
v = tf.Variable([1.0, 2.0], name = "v")
print(v)
print(id(v))
v.assign_add([1.0, 1.0])
print(v)
print(id(v))
```

**output**

```console
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([1., 2.], dtype=float32)>
5262300880
<tf.Variable 'v:0' shape=(2,) dtype=float32, numpy=array([2., 3.], dtype=float32)>
5262300880
```

