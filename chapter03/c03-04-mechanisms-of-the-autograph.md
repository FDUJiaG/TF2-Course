# Mechanisms of the AutoGraph

有三种计算图的构建方式
- 静态计算图
- 动态计算图
- Autograph

TensorFlow2.0 主要使用的是动态计算图和 Autograph
- 动态计算图易于调试，编码效率较高，但执行效率偏低
- 静态计算图执行效率很高，但较难调试
- Autograph 机制可以将动态图转换成静态计算图，兼收执行效率和编码效率之利

不过 Autograph 机制能够转换的代码并不是没有任何约束的，有一些编码规范需要遵循，否则可能会转换失败或者不符合预期

我们将着重介绍 Autograph 的编码规范和 Autograph 转换成静态图的原理，并介绍使用 `tf.Module` 来更好地构建 Autograph，本节我们介绍 Autograph 的机制原理

## The Mechanics of Autograph

```python
import tensorflow as tf
import numpy as np 

@tf.function(autograph=True)
def myadd(a, b):
    for i in tf.range(3):
        tf.print(i)
    c = a + b
    print("tracing")
    return c
```

后面什么都没有发生，仅仅是在 Python 堆栈中记录了这样一个函数的签名

当 **第一次调用** 这个被 `@tf.function` 装饰的函数时

```python
outer = myadd(tf.constant("hello"), tf.constant("world"))
```

**output**

```console
tracing
0
1
2
```

发生了 2 件事情

第一件事情是 **创建计算图**，即创建一个静态计算图，跟踪执行一遍函数体中的 Python 代码，确定各个变量的 Tensor 类型，并根据执行顺序将算子添加到计算图中

在这个过程中，如果开启了 `autograph=True` (默认开启)，会将 Python 控制流转换成 TensorFlow 图内控制流，主要是将 `if` 语句转换成 `tf.cond` 算子表达，将 `while` 和 `for` 循环语句转换成 `tf.while_loop` 算子表达，并在必要的时候添加 `tf.control_dependencies` 指定执行顺序依赖关系

相当于在 TensorFlow1.0 执行了类似下面的语句

```python
g = tf.Graph()
with g.as_default():
    a = tf.placeholder(shape=[], dtype=tf.string)
    b = tf.placeholder(shape=[], dtype=tf.string)
    cond = lambda i: i < tf.constant(3)
    def body(i):
        tf.print(i)
        return(i + 1)
    loop = tf.while_loop(cond, body, loop_vars=[0])
    loop
    with tf.control_dependencies(loop):
        c = tf.strings.join([a, b])
    print("tracing")
```

第二件事情是 **执行计算图**，相当于在 TensorFlow1.0 中执行了下面的语句

```python
with tf.Session(graph=g) as sess:
    sess.run(c, feed_dict={a: tf.constant("hello"), b: tf.constant("world")})
```

因此，先看到的是第一个步骤的结果，即 Python 调用标准输出流打印 `tracing` 语句，然后看到第二个步骤的结果，TensorFlow 调用标准输出流打印 `1, 2, 3`

当 **再次用相同的输入参数类型调用** 这个被 `@tf.function` 装饰的函数时

```python
outer = myadd(tf.constant("good"), tf.constant("morning"))
```

**output**

```console
0
1
2
```

只会发生一件事情，那就是上面步骤的第二步，执行计算图，所以这一次我们没有看到打印 `tracing` 的结果

当 **再次用不同的的输入参数类型调用** 这个被 `@tf.function` 装饰的函数时

```python
outer = myadd(tf.constant(1), tf.constant(2))
```

**output**

```console
tracing
0
1
2
```

由于输入参数的类型已经发生变化，已经创建的计算图不能够再次使用，需要重新做 2 件事情
- 创建新的计算图
- 执行计算图

所以重新看到两个步骤
- 先看到第一个步骤的结果即 Python 调用标准输出流打印 `tracing` 语句
- 然后看到第二个步骤的结果，TensorFlow 调用标准输出流打印 `1, 2, 3`

**注意** 如果调用被 `@tf.function` 装饰的函数，输入的参数不是 Tensor 类型，则每次都会 **重新创建计算图**

例如下面代码，两次都会重新创建计算图，因此，一般建议调用 `@tf.function` 时应传入 Tensor 类型

```python
outer = myadd("hello", "world")
outer = myadd("good", "morning")
```

**output**

```console
tracing
0
1
2
tracing
0
1
2
```

## Reinterpreting Autograph's Coding Specifications

了解了以上 Autograph 的机制原理，我们也就能够理解 Autograph 编码规范的 3 条建议了

1. 被 `@tf.function` 修饰的函数应尽量使用 TensorFlow 中的函数而不是 Python 中的其他函数，例如使用 `tf.print` 而不是 `print`
    
    【解释】 Python 中的函数仅仅会在跟踪执行函数以创建静态图的阶段使用，普通 Python 函数是无法嵌入到静态计算图中的，所以在计算图构建好之后再次调用的时候，这些 Python 函数并没有被计算，而 TensorFlow 中的函数则可以嵌入到计算图中，使用普通的Python 函数会导致被 `@tf.function` 修饰前 **eager 执行** 和被 `@tf.function` 修饰后 **静态图执行** 的输出不一致

1. 避免在 `@tf.function` 修饰的函数内部定义 `tf.Variable`

    【解释】 如果函数内部定义了 `tf.Variable`，那么在 **eager 执行** 时，这种创建 `tf.Variable` 的行为在每次函数调用时候都会发生，但是在 **静态图执行** 时，这种创建 `tf.Variable` 的行为只会发生在第一步跟踪 Python 代码逻辑创建计算图时，这会导致被 `@tf.function` 修饰前 **eager 执行** 和被 `@tf.function` 修饰后 **静态图执行** 输出不一致。实际上，TensorFlow 在这种情况下一般会报错。

1. 被 `@tf.function` 修饰的函数不可修改该函数外部的 Python 列表或字典等数据结构变量

    【解释】 静态计算图是被编译成 C++ 代码在 TensorFlow 内核中执行的，Python 中的列表和字典等数据结构变量是无法嵌入到计算图中，它们仅仅能够在创建计算图时被读取，在执行计算图时是无法修改 Python 中的列表或字典这样的数据结构变量的
