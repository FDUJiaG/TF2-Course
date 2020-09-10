# Structural Operations of the Tensor

张量结构操作诸如
- 张量创建
- 索引切片
- 维度变换
- 合并分割

## Tensor Creation

张量创建的许多方法和 Numpy 中创建 ndarray 的方法很像

```python
import tensorflow as tf
import numpy as np 
```

- 创建常数列表

```python
a = tf.constant([1, 2, 3], dtype=tf.float32)
tf.print(a)
```

**output**

```console
[1 2 3]
```

- 创建计数序列

```python
b = tf.range(1, 10, delta=2)
tf.print(b)
```

**output**

```console
[1 3 5 7 9]
```

- 创建等差数列

```python
c = tf.linspace(0.0, 2 * 3.14, 100)
tf.print(c)
```

**output**

```console
[0 0.0634343475 0.126868695 ... 6.15313148 6.21656609 6.28]
```

- 创建全 `0` 矩阵

```python
d = tf.zeros([3, 3])
tf.print(d)
```

**output**

```console
[[0 0 0]
 [0 0 0]
 [0 0 0]]
```

- 创建全 `1` 矩阵

```python
a = tf.ones([3, 3])
b = tf.zeros_like(a, dtype=tf.float32)
tf.print(a)
tf.print(b)
```

**output**

```console
[[1 1 1]
 [1 1 1]
 [1 1 1]]
[[0 0 0]
 [0 0 0]
 [0 0 0]]
```

- 矩阵元素填充

```python
b = tf.fill([3, 2], 5)
tf.print(b)
```

**output**

```console
[[5 5]
 [5 5]
 [5 5]]
```

- 随机生成均匀分布列

```python
tf.random.set_seed(1.0)
a = tf.random.uniform([5], minval=0, maxval=10)
tf.print(a)
```

**output**

```console
[1.65130854 9.01481247 6.30974197 4.34546089 2.9193902]
```

- 随机生成正态分布列

```python
b = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
tf.print(b)
```

**output**

```console
[[0.403087884 -1.0880208 -0.0630953535]
 [1.33655667 0.711760104 -0.489286453]
 [-0.764221311 -1.03724861 -1.25193381]]
```

- 随机生成正态分布列，并剔除 2 倍方差以外数据

```python
c = tf.random.truncated_normal((5, 5), mean=0.0, stddev=1.0, dtype=tf.float32)
tf.print(c)
```

**output**

```console
[[-0.457012236 -0.406867266 0.728577733 -0.892977774 -0.369404584]
 [0.323488563 1.19383323 0.888299048 1.25985599 -1.95951891]
 [-0.202244401 0.294496894 -0.468728036 1.29494202 1.48142183]
 [0.0810953453 1.63843894 0.556645 0.977199793 -1.17777884]
 [1.67368948 0.0647980496 -0.705142677 -0.281972528 0.126546144]]
```

- 特殊矩阵

```python
I = tf.eye(3, 3)                   # 单位阵
tf.print(I)
tf.print(" ")
t = tf.linalg.diag([1, 2, 3])      # 对角阵
tf.print(t)
```

**output**

```console
[[1 0 0]
 [0 1 0]
 [0 0 1]]
 
[[1 0 0]
 [0 2 0]
 [0 0 3]]
```

## Index Slice

张量的索引切片方式和 Numpy 几乎是一样的，切片时支持缺省参数和省略号

- 对于 `tf.Variable`，可以通过索引和切片对部分元素进行修改
- 对于提取张量的连续子区域，也可以使用 `tf.slice`
- 对于不规则的切片提取，可以使用 `tf.gather`， `tf.gather_nd`， `tf.boolean_mask`
       - `tf.boolean_mask` 功能最为强大，它可以实现 `tf.gather`， `tf.gather_nd` 的功能
       - 并且 `tf.boolean_mask` 还可以实现布尔索引

如果要通过修改张量的某些元素得到新的张量，可以使用 `tf.where`， `tf.scatter_nd`

### Rule Structure

- 首先来看一个二阶矩阵的例子

```python
tf.random.set_seed(3)
t = tf.random.uniform([5, 5], minval=0, maxval=10, dtype=tf.int32)
tf.print(t)
```

**output**

```console
[[4 7 4 2 9]
 [9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]
 [3 7 0 0 3]]
```

- 查看第 1 行和最后 1 行

```python
tf.print(t[0])
tf.print(t[-1])
```

**output**

```console
[4 7 4 2 9]
[3 7 0 0 3]
```

- 查看第 2 行第 4 列

```python
tf.print(t[1, 3])
tf.print(t[1][3])
```

**output**

```console
4
4
```

- 查看 2-4 行

```python
tf.print(t[1: 4, :])
tf.print(tf.slice(t, [1, 0], [3, 5]))     # tf.slice(input, begin_vector, size_vector)
```

**output**

```console
[[9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]]
[[9 1 2 4 7]
 [7 2 7 4 0]
 [9 6 9 7 2]]
```

- 查看 2-4 行，1 和 3 两列

```python
tf.print(t[1: 4, : 4: 2])
```

**output**

```console
[[9 2]
 [7 7]
 [9 9]]
```

- 对变量来说，还可以使用索引和切片修改部分元素

```python
x = tf.Variable([[1, 2], [3, 4]], dtype=tf.float32)
x[1, :].assign(tf.constant([0.0, 0.0]))
tf.print(x)
```

**output**

```console
[[1 2]
 [0 0]]
```

- 对于更高阶的矩阵

```python
a = tf.random.uniform([3, 3, 3], minval=0, maxval=10, dtype=tf.int32)
tf.print(a)
```

**output**

```console
[[[7 3 9]
  [9 0 7]
  [9 6 7]]

 [[1 3 3]
  [0 8 1]
  [3 1 0]]

 [[4 0 6]
  [6 2 2]
  [7 9 5]]]
```

- 省略号可以表示多个冒号

```python
tf.print(a[..., 1])
```

**output**

```console
[[3 0 6]
 [3 8 1]
 [0 2 9]]
```

### Irregular Structure

以上切片方式相对规则，对于不规则的切片提取，可以使用 `tf.gather`， `tf.gather_nd`， `tf.boolean_mask`

- 考虑班级成绩册的例子，有 4 个班级，每个班级 10 个学生，每个学生 7 门科目成绩。可以用一个 4×10×7 的张量来表示

```python
scores = tf.random.uniform((4, 10, 7), minval=0, maxval=100, dtype=tf.int32)
tf.print(scores)
```

**output**

```console
[[[52 82 66 ... 17 86 14]
  [8 36 94 ... 13 78 41]
  [77 53 51 ... 22 91 56]
  ...
  [11 19 26 ... 89 86 68]
  [60 72 0 ... 11 26 15]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [83 36 31 ... 75 38 85]
  [54 26 67 ... 60 68 98]
  ...
  [20 5 18 ... 32 45 3]
  [72 52 81 ... 88 41 20]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [78 71 54 ... 43 98 81]
  [21 66 53 ... 97 75 77]
  ...
  [6 74 3 ... 53 65 43]
  [98 36 72 ... 33 36 81]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [35 8 82 ... 11 59 97]
  [44 6 99 ... 81 60 27]
  ...
  [76 26 35 ... 51 8 17]
  [33 52 53 ... 78 37 31]
  [71 27 44 ... 0 52 16]]]
```

- 抽取每个班级第 1 个学生，第 6 个学生，第 10 个学生的全部成绩

```python
p = tf.gather(scores, [0, 5, 9], axis=1)
tf.print(p)
```

**output**

```console
[[[52 82 66 ... 17 86 14]
  [24 80 70 ... 72 63 96]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [46 10 94 ... 23 18 92]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [19 12 23 ... 87 86 25]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [6 41 79 ... 97 43 13]
  [71 27 44 ... 0 52 16]]]
```

- 抽取每个班级第 1 个学生，第 6 个学生，第 10 个学生的第 2 门课程，第 4 门课程，第 7 门课程成绩

```python
q = tf.gather(tf.gather(scores, [0, 5, 9], axis=1), [1, 3, 6], axis=2)
tf.print(q)
```

**output**

```console
[[[82 55 14]
  [80 46 96]
  [99 58 74]]

 [[73 48 81]
  [10 38 92]
  [21 86 90]]

 [[80 57 60]
  [12 34 25]
  [78 71 21]]

 [[57 75 3]
  [41 47 13]
  [27 96 16]]]
```

- 抽取第 1 个班级第 1 个学生，第 3 个班级的第 5 个学生，第 4 个班级的第 7 个学生的全部成绩

```python
# indices 的长度为采样样本的个数，每个元素为采样位置的坐标
s = tf.gather_nd(scores, indices=[(0, 0), (2, 4), (3, 6)])
s
```

**output**

```console
<tf.Tensor: shape=(3, 7), dtype=int32, numpy=
array([[52, 82, 66, 55, 17, 86, 14],
       [99, 94, 46, 70,  1, 63, 41],
       [46, 83, 70, 80, 90, 85, 17]], dtype=int32)>
```

- 以上 `tf.gather` 和 `tf.gather_nd` 的功能也可以用 `tf.boolean_mask` 来实现，比如抽取每个班级第 1 个学生，第 6 个学生，第 10 个学生的全部成绩

```python
p = tf.boolean_mask(scores, [True, False, False, False, False, 
                            True, False, False, False, True], axis=1)
tf.print(p)
```

**output**

```console
[[[52 82 66 ... 17 86 14]
  [24 80 70 ... 72 63 96]
  [24 99 38 ... 97 44 74]]

 [[79 73 73 ... 35 3 81]
  [46 10 94 ... 23 18 92]
  [0 21 89 ... 53 10 90]]

 [[52 80 22 ... 29 25 60]
  [19 12 23 ... 87 86 25]
  [61 78 70 ... 7 59 21]]

 [[56 57 45 ... 23 15 3]
  [6 41 79 ... 97 43 13]
  [71 27 44 ... 0 52 16]]]
```

- 抽取第 1 个班级第 1 个学生，第 3 个班级的第 5 个学生，第 4 个班级的第 7 个学生的全部成绩

```python
s = tf.boolean_mask(scores,
    [[True, False, False, False, False, False, False, False, False, False],
     [False, False, False, False, False, False, False, False, False, False],
     [False, False, False, False, True, False, False, False, False, False],
     [False, False, False, False, False, False, True, False, False, False]])
tf.print(s)
```

**output**

```console
[[52 82 66 ... 17 86 14]
 [99 94 46 ... 1 63 41]
 [46 83 70 ... 90 85 17]]
```

- 利用 tf.boolean_mask 可以实现布尔索引，比如找到矩阵中小于 0 的元素

```python
c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
tf.print(c, '\n')

tf.print(tf.boolean_mask(c, c < 0)) 
tf.print(c[c < 0])   # 布尔索引
```

**output**

```console
[[-1 1 -1]
 [2 2 -2]
 [3 -3 3]]

[-1 -1 -2 -3]
[-1 -1 -2 -3]
```

### Get New Tensor

以上这些方法仅能提取张量的部分元素值，但不能更改张量的部分元素值得到新的张量，如果要通过修改张量的部分元素值得到新的张量，可以使用 `tf.where` 和 `tf.scatter_nd`

- `tf.where` 可以理解为 `if` 的张量版本，此外它还可以用于找到满足条件的所有元素的位置坐标
- `tf.scatter_nd` 的作用和 `tf.gather_nd` 有些相反，`tf.gather_nd` 用于收集张量的给定位置的元素，而 `tf.scatter_nd` 可以将某些值插入到一个给定 `shape` 的全 `0` 的张量的指定位置处

可以看一个实例
- 找到张量中小于 `0` 的元素,将其换成 `np.nan` 得到新的张量

```python
c = tf.constant([[-1, 1, -1], [2, 2, -2], [3, -3, 3]], dtype=tf.float32)
d = tf.where(c < 0, tf.fill(c.shape, np.nan), c)
d
```

**output**

```console
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[nan,  1., nan],
       [ 2.,  2., nan],
       [ 3., nan,  3.]], dtype=float32)>
```

- 如果 where 只有一个参数，将返回所有满足条件的位置坐标

```python
indices = tf.where(c < 0)
indices
```

**output**

```console
<tf.Tensor: shape=(4, 2), dtype=int64, numpy=
array([[0, 0],
       [0, 2],
       [1, 2],
       [2, 1]])>
```

- 将张量的第 `[0,0]` 和 `[2,1]` 两个位置元素替换为 `0` 得到新的张量

```python
d = c - tf.scatter_nd([[0, 0], [2, 1]], [c[0, 0], c[2, 1]], c.shape)
d
```

**output**

```console
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[ 0.,  1., -1.],
       [ 2.,  2., -2.],
       [ 3.,  0.,  3.]], dtype=float32)>
```

- `scatter_nd` 的作用和 `gather_nd` 有些相反，可以将某些值插入到一个给定 shape 的全 `0` 的张量的指定位置处

```python
indices = tf.where(c < 0)
tf.scatter_nd(indices, tf.gather_nd(c, indices), c.shape)
```

**output**

```console
<tf.Tensor: shape=(3, 3), dtype=float32, numpy=
array([[-1.,  0., -1.],
       [ 0.,  0., -2.],
       [ 0., -3.,  0.]], dtype=float32)>
```

## Dimensional Transformation

维度变换相关函数主要有 `tf.reshape`， `tf.squeeze`， `tf.expand_dims`， `tf.transpose`

- `tf.reshape` 可以改变张量的形状
- `tf.squeeze` 可以减少维度
- `tf.expand_dims` 可以增加维度
- `tf.transpose` 可以交换维度

### Reshape the Tensor

`tf.reshape` 可以改变张量的形状，但是其本质上不会改变张量元素的存储顺序，所以，该操作实际上非常迅速，并且是可逆的

```python
a = tf.random.uniform(shape=[1, 3, 3, 2],
                      minval=0, maxval=255, dtype=tf.int32)
tf.print(a.shape)
tf.print(a)
```

**output**

```console
TensorShape([1, 3, 3, 2])
[[[[135 178]
   [26 116]
   [29 224]]

  [[179 219]
   [153 209]
   [111 215]]

  [[39 7]
   [138 129]
   [59 205]]]]
```

- 改成 `(3, 6)` 形状的张量

```python
b = tf.reshape(a, [3, 6])
tf.print(b.shape)
tf.print(b)
```

**output**

```console
TensorShape([3, 6])
[[135 178 26 116 29 224]
 [179 219 153 209 111 215]
 [39 7 138 129 59 205]]
```

- 改回成 `[1, 3, 3, 2]` 形状的张量


```python
c = tf.reshape(b, [1, 3, 3, 2])
tf.print(c)
```

**output**

```console
[[[[135 178]
   [26 116]
   [29 224]]

  [[179 219]
   [153 209]
   [111 215]]

  [[39 7]
   [138 129]
   [59 205]]]]
```

### Dimension Reduction

如果张量在某个维度上只有一个元素，利用 `tf.squeeze` 可以消除这个维度，和 `tf.reshape` 相似，它本质上不会改变张量元素的存储顺序

张量的各个元素在内存中是线性存储的，其一般规律是，同一层级中的相邻元素的物理地址也相邻

```python
s = tf.squeeze(a)
tf.print(s.shape)
tf.print(s)
```

**output**

```console
TensorShape([3, 3, 2])
[[[135 178]
  [26 116]
  [29 224]]

 [[179 219]
  [153 209]
  [111 215]]

 [[39 7]
  [138 129]
  [59 205]]]
```

### Adding Dimensions

```python
d = tf.expand_dims(s, axis=0)      # 在第 0 维插入长度为 1 的一个维度
d
```

**output**

```console
<tf.Tensor: shape=(1, 3, 3, 2), dtype=int32, numpy=
array([[[[135, 178],
         [ 26, 116],
         [ 29, 224]],

        [[179, 219],
         [153, 209],
         [111, 215]],

        [[ 39,   7],
         [138, 129],
         [ 59, 205]]]], dtype=int32)>
```

### Transpose Dimension

- `tf.transpose` 可以交换张量的维度，与 `tf.reshape` 不同，它会改变张量元素的存储顺序
- `tf.transpose` 常用于图片存储格式的变换上

```python
# Batch, Height, Width, Channel
a = tf.random.uniform(shape=[100, 600, 600, 4], minval=0, maxval=255, dtype=tf.int32)
tf.print(a.shape)

# 转换成 Channel, Height, Width, Batch
s= tf.transpose(a,perm=[3, 1, 2, 0])
tf.print(s.shape)
```

**output**

```console
TensorShape([100, 600, 600, 4])
TensorShape([4, 600, 600, 100])
```

## Merger and Division

和 Numpy 类似，可以用 `tf.concat` 和 `tf.stack` 方法对多个张量进行合并，可以用 `tf.split` 方法把一个张量分割成多个张量

`tf.concat` 和 `tf.stack` 有略微的区别，`tf.concat` 是连接，不会增加维度，而 `tf.stack` 是堆叠，会增加维度

### Concat the Tensor

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.constant([[9.0, 10.0], [11.0, 12.0]])

tf.concat([a, b, c], axis=0)
```

**output**

```console
<tf.Tensor: shape=(6, 2), dtype=float32, numpy=
array([[ 1.,  2.],
       [ 3.,  4.],
       [ 5.,  6.],
       [ 7.,  8.],
       [ 9., 10.],
       [11., 12.]], dtype=float32)>
```

- 横向连接

```python
tf.concat([a, b, c], axis=1)
```

**output**

```console
<tf.Tensor: shape=(2, 6), dtype=float32, numpy=
array([[ 1.,  2.,  5.,  6.,  9., 10.],
       [ 3.,  4.,  7.,  8., 11., 12.]], dtype=float32)>
```

### Stack the Tensor

```python
tf.stack([a, b, c])
```

**output**

```console
<tf.Tensor: shape=(3, 2, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 3.,  4.]],

       [[ 5.,  6.],
        [ 7.,  8.]],

       [[ 9., 10.],
        [11., 12.]]], dtype=float32)>
```

- 横向堆叠

```python
tf.stack([a, b, c], axis=1)
```

**output**

```console
<tf.Tensor: shape=(2, 3, 2), dtype=float32, numpy=
array([[[ 1.,  2.],
        [ 5.,  6.],
        [ 9., 10.]],

       [[ 3.,  4.],
        [ 7.,  8.],
        [11., 12.]]], dtype=float32)>
```

### Split the Tensor

```python
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
c = tf.constant([[9.0, 10.0], [11.0, 12.0]])

c = tf.concat([a, b, c], axis=0)
```

`tf.split` 是 `tf.concat` 的逆运算，可以指定分割份数平均分割，也可以通过指定每份的记录数量进行分割

```python
# tf.split(value, num_or_size_splits, axis)
tf.split(c, 3, axis=0)  # 指定分割份数，平均分割
```

**output**

```console
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[5., 6.],
        [7., 8.]], dtype=float32)>,
 <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[ 9., 10.],
        [11., 12.]], dtype=float32)>]
```

- 指定每份的分割数量

```python
tf.split(c, [2, 1, 3], axis=0)
```

**output**

```console
[<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
 array([[1., 2.],
        [3., 4.]], dtype=float32)>,
 <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[5., 6.]], dtype=float32)>,
 <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
 array([[ 7.,  8.],
        [ 9., 10.],
        [11., 12.]], dtype=float32)>]
```
