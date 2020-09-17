# 05-02 Three Ways of Training

模型的训练主要方法有

- 内置 `fit` 方法
- 内置 `tran_on_batch` 方法
- 自定义训练循环

**注意** `fit_generator` 方法在 `tf.keras` 中不推荐使用，其功能已经被 `fit` 包含

## Data Preparation

打印时间分割线函数

```python
import numpy as np 
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import * 

# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)

    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)

    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}", m))==1:
            return(tf.strings.format("0{}", m))
        else:
            return(tf.strings.format("{}", m))
    
    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                timeformat(second)], separator=":")
    tf.print("==========" * 8 + timestring)
```

构建数据管道

```python
MAX_LEN = 300
BATCH_SIZE = 32
(x_train, y_train), (x_test, y_test) = datasets.reuters.load_data()
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=MAX_LEN)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=MAX_LEN)

MAX_WORDS = x_train.max() + 1
CAT_NUM = y_train.max() + 1

ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()
   
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)) \
          .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
          .prefetch(tf.data.experimental.AUTOTUNE).cache()
```

## Built-in fit Method

该方法功能非常强大，支持对 Numpy Array，`tf.data.Dataset` 以及 Python Generator 数据进行训练，并且可以通过设置回调函数实现对训练过程的复杂控制逻辑

```python
tf.keras.backend.clear_session()
def create_model():
    
    model = models.Sequential()
    model.add(layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM, activation="softmax"))
    return(model)

def compile_model(model):
    model.compile(
        optimizer=optimizers.Nadam(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[
            metrics.SparseCategoricalAccuracy(), 
            metrics.SparseTopKCategoricalAccuracy(5)
        ]
    )
    return(model)
 
model = create_model()
model.summary()
model = compile_model(model)
```

**output**

```console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 300, 7)            216874
_________________________________________________________________
conv1d (Conv1D)              (None, 296, 64)           2304
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 148, 64)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 146, 32)           6176
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 73, 32)            0
_________________________________________________________________
flatten (Flatten)            (None, 2336)              0
_________________________________________________________________
dense (Dense)                (None, 46)                107502
=================================================================
Total params: 332,856
Trainable params: 332,856
Non-trainable params: 0
_________________________________________________________________
```

训练模型

```python
history = model.fit(ds_train, validation_data=ds_test, epochs=10)
```

**output**

```console
Epoch 1/10
281/281 [==============================] - 8s 28ms/step - loss: 2.0323 - sparse_categorical_accuracy: 0.4692 - sparse_top_k_categorical_accuracy: 0.7444 - val_loss: 1.6836 - val_sparse_categorical_accuracy: 0.5592 - val_sparse_top_k_categorical_accuracy: 0.7560
Epoch 2/10
281/281 [==============================] - 10s 34ms/step - loss: 1.5015 - sparse_categorical_accuracy: 0.6143 - sparse_top_k_categorical_accuracy: 0.7936 - val_loss: 1.5454 - val_sparse_categorical_accuracy: 0.6158 - val_sparse_top_k_categorical_accuracy: 0.7836
Epoch 3/10
281/281 [==============================] - 7s 25ms/step - loss: 1.2051 - sparse_categorical_accuracy: 0.6887 - sparse_top_k_categorical_accuracy: 0.8526 - val_loss: 1.5315 - val_sparse_categorical_accuracy: 0.6380 - val_sparse_top_k_categorical_accuracy: 0.8085
Epoch 4/10
281/281 [==============================] - 7s 25ms/step - loss: 0.9343 - sparse_categorical_accuracy: 0.7575 - sparse_top_k_categorical_accuracy: 0.9092 - val_loss: 1.7171 - val_sparse_categorical_accuracy: 0.6278 - val_sparse_top_k_categorical_accuracy: 0.8072
Epoch 5/10
281/281 [==============================] - 7s 24ms/step - loss: 0.7034 - sparse_categorical_accuracy: 0.8208 - sparse_top_k_categorical_accuracy: 0.9458 - val_loss: 1.9922 - val_sparse_categorical_accuracy: 0.6211 - val_sparse_top_k_categorical_accuracy: 0.8010
Epoch 6/10
281/281 [==============================] - 7s 26ms/step - loss: 0.5343 - sparse_categorical_accuracy: 0.8688 - sparse_top_k_categorical_accuracy: 0.9668 - val_loss: 2.2429 - val_sparse_categorical_accuracy: 0.6175 - val_sparse_top_k_categorical_accuracy: 0.8001
Epoch 7/10
281/281 [==============================] - 7s 23ms/step - loss: 0.4268 - sparse_categorical_accuracy: 0.9004 - sparse_top_k_categorical_accuracy: 0.9784 - val_loss: 2.5072 - val_sparse_categorical_accuracy: 0.6247 - val_sparse_top_k_categorical_accuracy: 0.7947
Epoch 8/10
281/281 [==============================] - 6s 23ms/step - loss: 0.3586 - sparse_categorical_accuracy: 0.9185 - sparse_top_k_categorical_accuracy: 0.9834 - val_loss: 2.7018 - val_sparse_categorical_accuracy: 0.6220 - val_sparse_top_k_categorical_accuracy: 0.7921
Epoch 9/10
281/281 [==============================] - 6s 23ms/step - loss: 0.3116 - sparse_categorical_accuracy: 0.9281 - sparse_top_k_categorical_accuracy: 0.9881 - val_loss: 2.8627 - val_sparse_categorical_accuracy: 0.6207 - val_sparse_top_k_categorical_accuracy: 0.7934
Epoch 10/10
281/281 [==============================] - 7s 23ms/step - loss: 0.2758 - sparse_categorical_accuracy: 0.9343 - sparse_top_k_categorical_accuracy: 0.9915 - val_loss: 3.0145 - val_sparse_categorical_accuracy: 0.6211 - val_sparse_top_k_categorical_accuracy: 0.7903
```

## Built-in train_on_batch Method

该内置方法相比较 `fit` 方法更加灵活，可以不通过回调函数而直接在批次层次上更加精细地控制训练的过程

```python
tf.keras.backend.clear_session()

def create_model():
    model = models.Sequential()

    model.add(layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM, activation="softmax"))
    return(model)

def compile_model(model):
    model.compile(
        optimizer=optimizers.Nadam(),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[
            metrics.SparseCategoricalAccuracy(), 
            metrics.SparseTopKCategoricalAccuracy(5)
        ]
    )
    return(model)
 
model = create_model()
model.summary()
model = compile_model(model)
```

**output**

```console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 300, 7)            216874
_________________________________________________________________
conv1d (Conv1D)              (None, 296, 64)           2304
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 148, 64)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 146, 32)           6176
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 73, 32)            0
_________________________________________________________________
flatten (Flatten)            (None, 2336)              0
_________________________________________________________________
dense (Dense)                (None, 46)                107502
=================================================================
Total params: 332,856
Trainable params: 332,856
Non-trainable params: 0
_________________________________________________________________
```

编写 `train_on_batch` 方法训练的函数，在后期降低学习率

```python
import json

def train_model(model, ds_train, ds_valid, epochs):

    for epoch in tf.range(1, epochs + 1):
        model.reset_metrics()

        # 在后期降低学习率
        if epoch == 5:
            model.optimizer.lr.assign(model.optimizer.lr/2.0)
            tf.print("\n[INFO] Lowering optimizer Learning Rate...\n")

        for x, y in ds_train:
            train_result = model.train_on_batch(x, y)

        for x, y in ds_valid:
            valid_result = model.test_on_batch(x, y, reset_metrics=False)

        if epoch % 1 ==0:
            printbar()
            tf.print("Epoch ", epoch, '/', epochs)
            tf.print("train:", json.dumps(
                dict(zip(model.metrics_names, train_result)), indent=4))
            tf.print("valid:", json.dumps(
                dict(zip(model.metrics_names, valid_result)), indent=4))
```

使用 `train_on_batch` 方法进行训练

```python
train_model(model, ds_train, ds_test, 10)
```

**output**

```console
================================================================================10:49:18
Epoch  1 / 10
train: {
    "loss": 0.8378490209579468,
    "sparse_categorical_accuracy": 0.6818181872367859,
    "sparse_top_k_categorical_accuracy": 0.9545454382896423
}
valid: {
    "loss": 1.5069658756256104,
    "sparse_categorical_accuracy": 0.645146906375885,
    "sparse_top_k_categorical_accuracy": 0.8054319024085999
}
================================================================================10:49:26
Epoch  2 / 10
train: {
    "loss": 0.4498234987258911,
    "sparse_categorical_accuracy": 0.9090909361839294,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 1.6743230819702148,
    "sparse_categorical_accuracy": 0.6362422108650208,
    "sparse_top_k_categorical_accuracy": 0.8040961623191833
}
================================================================================10:49:34
Epoch  3 / 10
train: {
    "loss": 0.2919714152812958,
    "sparse_categorical_accuracy": 0.9090909361839294,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 1.9409838914871216,
    "sparse_categorical_accuracy": 0.6335707902908325,
    "sparse_top_k_categorical_accuracy": 0.8036509156227112
}
================================================================================10:49:42
Epoch  4 / 10
train: {
    "loss": 0.17916302382946014,
    "sparse_categorical_accuracy": 0.9545454382896423,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.1986467838287354,
    "sparse_categorical_accuracy": 0.6322351098060608,
    "sparse_top_k_categorical_accuracy": 0.8054319024085999
}

[INFO] Lowering optimizer Learning Rate...

================================================================================10:49:50
Epoch  5 / 10
train: {
    "loss": 0.08607669919729233,
    "sparse_categorical_accuracy": 1.0,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.371387004852295,
    "sparse_categorical_accuracy": 0.6420302987098694,
    "sparse_top_k_categorical_accuracy": 0.8103294968605042
}
================================================================================10:49:57
Epoch  6 / 10
train: {
    "loss": 0.0642002746462822,
    "sparse_categorical_accuracy": 1.0,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.503800630569458,
    "sparse_categorical_accuracy": 0.6406945586204529,
    "sparse_top_k_categorical_accuracy": 0.8067675828933716
}
================================================================================10:50:05
Epoch  7 / 10
train: {
    "loss": 0.04979919269680977,
    "sparse_categorical_accuracy": 1.0,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.624760627746582,
    "sparse_categorical_accuracy": 0.6415850520133972,
    "sparse_top_k_categorical_accuracy": 0.8085485100746155
}
================================================================================10:50:12
Epoch  8 / 10
train: {
    "loss": 0.040903713554143906,
    "sparse_categorical_accuracy": 1.0,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.742506980895996,
    "sparse_categorical_accuracy": 0.6375778913497925,
    "sparse_top_k_categorical_accuracy": 0.8058770895004272
}
================================================================================10:50:20
Epoch  9 / 10
train: {
    "loss": 0.03392982482910156,
    "sparse_categorical_accuracy": 1.0,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.846313714981079,
    "sparse_categorical_accuracy": 0.638913631439209,
    "sparse_top_k_categorical_accuracy": 0.8058770895004272
}
================================================================================10:50:27
Epoch  10 / 10
train: {
    "loss": 0.030621279031038284,
    "sparse_categorical_accuracy": 1.0,
    "sparse_top_k_categorical_accuracy": 1.0
}
valid: {
    "loss": 2.9391837120056152,
    "sparse_categorical_accuracy": 0.6344612836837769,
    "sparse_top_k_categorical_accuracy": 0.8054319024085999
}
```

## Custom Training Loops

自定义训练循环无需编译模型，直接利用优化器根据损失函数反向传播迭代参数，拥有最高的灵活性

```python
tf.keras.backend.clear_session()

def create_model():
    
    model = models.Sequential()

    model.add(layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN))
    model.add(layers.Conv1D(filters=64, kernel_size=5, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Conv1D(filters=32, kernel_size=3, activation="relu"))
    model.add(layers.MaxPool1D(2))
    model.add(layers.Flatten())
    model.add(layers.Dense(CAT_NUM, activation="softmax"))
    return(model)

model = create_model()
model.summary()
```

**output**

```console
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 300, 7)            216874
_________________________________________________________________
conv1d (Conv1D)              (None, 296, 64)           2304
_________________________________________________________________
max_pooling1d (MaxPooling1D) (None, 148, 64)           0
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 146, 32)           6176
_________________________________________________________________
max_pooling1d_1 (MaxPooling1 (None, 73, 32)            0
_________________________________________________________________
flatten (Flatten)            (None, 2336)              0
_________________________________________________________________
dense (Dense)                (None, 46)                107502
=================================================================
Total params: 332,856
Trainable params: 332,856
Non-trainable params: 0
_________________________________________________________________
```

使用自定义的方法进行训练

```python
optimizer = optimizers.Nadam()
loss_func = losses.SparseCategoricalCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(model, features, labels):
    with tf.GradientTape() as tape:
        predictions = model(features, training=True)
        loss = loss_func(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss.update_state(loss)
    train_metric.update_state(labels, predictions)

@tf.function
def valid_step(model, features, labels):
    predictions = model(features)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):
        for features, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)

        logs = 'Epoch {}/{}, Loss:{}, Accuracy:{}, Valid Loss:{}, Valid Accuracy:{}'

        if epoch % 1 == 0:
            printbar()
            tf.print(tf.strings.format(
                logs, (epoch, epochs, train_loss.result(), train_metric.result(),
                valid_loss.result(), valid_metric.result())))

        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model, ds_train, ds_test, 10)
```

**output**

```console
================================================================================11:31:41
Epoch 1/10, Loss:0.176203713, Accuracy:0.948897779, Valid Loss:3.70648241, Valid Accuracy:0.587266266
================================================================================11:31:48
Epoch 2/10, Loss:0.165467, Accuracy:0.94900912, Valid Loss:3.77091956, Valid Accuracy:0.581032932
================================================================================11:31:57
Epoch 3/10, Loss:0.154677972, Accuracy:0.95045644, Valid Loss:3.87373424, Valid Accuracy:0.575244904
================================================================================11:32:07
Epoch 4/10, Loss:0.146738917, Accuracy:0.950901806, Valid Loss:3.9760251, Valid Accuracy:0.565449715
================================================================================11:32:17
Epoch 5/10, Loss:0.140053421, Accuracy:0.952015162, Valid Loss:4.12776613, Valid Accuracy:0.560997307
================================================================================11:32:29
Epoch 6/10, Loss:0.134411231, Accuracy:0.951681137, Valid Loss:4.29239607, Valid Accuracy:0.55921638
================================================================================11:32:35
Epoch 7/10, Loss:0.129114911, Accuracy:0.954130471, Valid Loss:4.44853115, Valid Accuracy:0.555654526
================================================================================11:32:43
Epoch 8/10, Loss:0.12188869, Accuracy:0.954909801, Valid Loss:4.67705631, Valid Accuracy:0.55654496
================================================================================11:32:50
Epoch 9/10, Loss:0.118974648, Accuracy:0.955243826, Valid Loss:4.86845875, Valid Accuracy:0.547640264
================================================================================11:32:56
Epoch 10/10, Loss:0.112785935, Accuracy:0.957804501, Valid Loss:5.05170345, Valid Accuracy:0.539626
```
