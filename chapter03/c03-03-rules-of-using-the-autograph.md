# Rules of Using the AutoGraph

有三种计算图的构建方式
- 静态计算图
- 动态计算图
- Autograph

TensorFlow2.0 主要使用的是动态计算图和 Autograph
- 动态计算图易于调试，编码效率较高，但执行效率偏低
- 静态计算图执行效率很高，但较难调试
- Autograph 机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利

不过 Autograph 机制能够转换的代码并不是没有任何约束的，有一些编码规范需要遵循，否则可能会转换失败或者不符合预期

我们将着重介绍 Autograph 的编码规范和 Autograph 转换成静态图的原理，并介绍使用 `tf.Module` 来更好地构建 Autograph，我们首先从 Autograph 的编码规范开始

## Summary of Autograph Coding Specifications

1. 被 `@tf.function` 修饰的函数应尽可能使用 TensorFlow 中的函数而不是 Python 中的其他函数。例如使用 `tf.print` 而不是 `print`，使用 `tf.range` 而不是 `range`，使用 `tf.constant(True)` 而不是 `True`
1. 避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`
1. 被 `@tf.function` 修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量

## Autograph Encoding Specification Explanation

### Use TensorFlow's Internal Functions whenever Possible

```python
import numpy as np
import tensorflow as tf

@tf.function
def np_random():
    a = np.random.randn(3, 3)
    tf.print(a)
    
@tf.function
def tf_random():
    a = tf.random.normal((3, 3))
    tf.print(a)
```

- `np_random` 每次执行都是一样的结果

```python
np_random()
np_random()
```

**output**

```console
array([[ 1.19099038,  0.20530001, -0.25965183],
       [ 1.48162286,  0.9139405 ,  1.25573669],
       [ 0.12049997,  0.49903263,  1.02769522]])
array([[ 1.19099038,  0.20530001, -0.25965183],
       [ 1.48162286,  0.9139405 ,  1.25573669],
       [ 0.12049997,  0.49903263,  1.02769522]])
```

- `tf_random` 每次执行都会有重新生成随机数

```python
tf_random()
tf_random()
```

**output**

```console
[[-0.945860744 -0.033374086 0.344509631]
 [-0.59630847 1.10220349 0.587554097]
 [0.0221184 0.178448856 0.109851979]]
[[0.544216216 0.138761416 0.0195933841]
 [-0.84833473 0.383355081 -1.67035949]
 [0.207884297 -0.729542911 -1.05002272]]
```

### Avoid Defining Variables Inside the Modified Function

```python
x = tf.Variable(1.0, dtype=tf.float32)
@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return(x)

o1 = outer_var()
o2 = outer_var()
```

**output**

```console
2
3
```

避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`，否则可能会报错

```python
@tf.function
def inner_var():
    x = tf.Variable(1.0,dtype = tf.float32)
    x.assign_add(1.0)
    tf.print(x)
    return(x)

# 执行将报错
# i1 = inner_var()
# i2 = inner_var()
```

**output**

```console
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-10-a89269e2b6b4> in <module>
      7
      8 # 执行将报错
----> 9 i1 = inner_var()
     10 i2 = inner_var()

~/anaconda3/envs/[ENV_NAME]/lib/[PYTHON_VERSION]/site-packages/tensorflow/python/eager/def_function.py in __call__(self, *args, **kwds)
    778       else:
    779         compiler = "nonXla"
--> 780         result = self._call(*args, **kwds)
    781
    782       new_tracing_count = self._get_tracing_count()

......

ValueError: tf.function-decorated function tried to create variables on non-first call.
```

### The Modified Func cannot Modify Structure Type Vars External to the Func

被 `@tf.function` 修饰的函数不可修改该函数外部的 Python 列表或字典等结构类型变量

- 不加 `@tf.function` 时，可以修改外部列表

```python
tensor_list = []

# @tf.function  # 加上这一行切换成 Autograph结果将不符合预期！！！
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
```

**output**

```console
[<tf.Tensor: shape=(), dtype=float32, numpy=5.0>, <tf.Tensor: shape=(), dtype=float32, numpy=6.0>]
```

- 增加 `@tf.function` 时，不可以修改外部列表

```python
tensor_list = []

@tf.function    # 加上这一行切换成 Autograph 结果将不符合预期！！！
def append_tensor(x):
    tensor_list.append(x)
    return tensor_list

append_tensor(tf.constant(5.0))
append_tensor(tf.constant(6.0))
print(tensor_list)
```

**output**

```console
[<tf.Tensor 'x:0' shape=() dtype=float32>]
```
