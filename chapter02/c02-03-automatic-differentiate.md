# Automatic Differentiate

神经网络通常依赖反向传播求梯度来更新网络参数，求梯度过程通常是一件非常复杂而容易出错的事情

而深度学习框架可以帮助我们自动地完成这种求梯度运算

Tensorflow一般使用梯度磁带tf.GradientTape来记录正向运算过程，然后反播磁带自动得到梯度值

这种利用tf.GradientTape求微分的方法叫做Tensorflow的自动微分机制

## Derivatives using Gradient Tape

### Derivation of the Independent Variable

```python
import tensorflow as tf
import numpy as np 

# f(x) = a * x ** 2 + b * x + c 的导数

x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

with tf.GradientTape() as tape:
    y = a * tf.pow(x, 2) + b * x + c
    
dy_dx = tape.gradient(y, x)
print(dy_dx)
```

**output**

```console
tf.Tensor(-2.0, shape=(), dtype=float32)
```

### Derivation of a Constant

```python
# 对常量张量也可以求导，需要增加 watch
with tf.GradientTape() as tape:
    tape.watch([a, b, c])
    y = a*tf.pow(x, 2) + b * x + c
    
dy_dx, dy_da, dy_db, dy_dc = tape.gradient(y, [x, a, b, c])
print(dy_da)
print(dy_dc)
```

**output**

```console
tf.Tensor(0.0, shape=(), dtype=float32)
tf.Tensor(1.0, shape=(), dtype=float32)
```

### Derivative of the Second Order

```python
# 可以求二阶导数
with tf.GradientTape() as tape2:
    with tf.GradientTape() as tape1:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape1.gradient(y, x)
dy2_dx2 = tape2.gradient(dy_dx, x)

print(dy2_dx2)
```

**output**

```console
tf.Tensor(2.0, shape=(), dtype=float32)
```

### Using in Autograph

```python
# 可以在 autograph 中使用
@tf.function
def f(x):
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    
    # 自变量转换成 tf.float32
    x = tf.cast(x, tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = a*tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    
    return((dy_dx,y))

tf.print(f(tf.constant(0.0)))
tf.print(f(tf.constant(1.0)))
```

**output**

```console
(-2, 1)
(0, 0)
```

## Find Minimum Values

### Using Gradient Tapes and Optimizers

- 使用 `optimizer.apply_gradients`

```python
# 求 f(x) = a * x ** 2 + b * x + c 的最小值
x = tf.Variable(0.0, name="x", dtype=tf.float32)
a = tf.constant(1.0)
b = tf.constant(-2.0)
c = tf.constant(1.0)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for _ in range(1000):
    with tf.GradientTape() as tape:
        y = a * tf.pow(x, 2) + b * x + c
    dy_dx = tape.gradient(y, x)
    optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])
    
tf.print("y =", y, "; x =", x)
```

**output**

```console
y = 0 ; x = 0.999998569
```

- 使用 `optimizer.minimize`

```python
# 求 f(x) = a * x ** 2 + b * x + c 的最小值
# optimizer.minimize 相当于先用 tape 求 gradient，再 apply_gradient
x = tf.Variable(0.0, name="x", dtype=tf.float32)

# 注意 f() 无参数
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return(y)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   
for _ in range(1000):
    optimizer.minimize(f, [x])
    
tf.print("y =", f(), "; x =", x)
```

```console
y = 0 ; x = 0.999998569
```

### Using Autograph and Optimizers

- 使用 `optimizer.apply_gradients`

```python
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

@tf.function
def minimizef():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)

    for _ in tf.range(1000):    # 注意 autograph 时使用 tf.range(1000) 而不是 range(1000)
        with tf.GradientTape() as tape:
            y = a * tf.pow(x, 2) + b * x + c
        dy_dx = tape.gradient(y, x)
        optimizer.apply_gradients(grads_and_vars=[(dy_dx, x)])

    y = a * tf.pow(x, 2) + b * x + c
    return y

tf.print(minimizef())
tf.print(x)
```

**output**

```console
0
0.999998569
```

- 使用 `optimizer.minimize`

```python
x = tf.Variable(0.0, name="x", dtype=tf.float32)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)   

@tf.function
def f():
    a = tf.constant(1.0)
    b = tf.constant(-2.0)
    c = tf.constant(1.0)
    y = a * tf.pow(x, 2) + b * x + c
    return(y)

@tf.function
def train(epoch):  
    for _ in tf.range(epoch):  
        optimizer.minimize(f, [x])
    return(f())

tf.print(train(1000))
tf.print(x)
```

**output**

```console
0
0.999998569
```

