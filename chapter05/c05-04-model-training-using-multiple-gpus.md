# 05-04 Model Training Using Multiple GPUs

如果使用多 GPU 训练模型，推荐使用内置 `fit` 方法，较为方便，仅需添加 2 行代码

## GPU Settings

详见上一节内容，首先登陆 [Google Drive](https://drive.google.com/drive/)，在 Colab 笔记本中，依次点击 **修改/笔记本设置**，在 **硬件加速器** 中选择 **GPU**

### MirroredStrategy Process Overview

- 训练开始前，该策略在所有 N 个计算设备上均各复制一份完整的模型
- 每次训练传入一个批次的数据时，将数据分成 N 份，分别传入 N 个计算设备（即数据并行）
- N 个计算设备使用本地变量（镜像变量）分别计算自己所获得的部分数据的梯度
- 使用分布式计算的 All-Reduce 操作，在计算设备间高效交换梯度数据并进行求和，使得最终每个设备都有了所有设备的梯度之和
- 使用梯度求和的结果更新本地变量（镜像变量）
- 当所有设备均更新本地变量后，进行下一轮训练（即该并行策略是同步的）

### Multiple GPUs Settings

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import * 
```

**output**

```console
2.3.0
```

此处在 Colab 上使用 1 个 GPU 模拟出 2 个逻辑 GPU 进行多 GPU 训练，如果实际有多个物理的 GPU 则更为方便，还可以把 VGPU 的设置删减掉

```python
# 此处在 Colab 上使用 1 个 GPU 模拟出 2 个逻辑 GPU 进行多 GPU 训练
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # 设置两个逻辑 GPU 模拟多 GPU 训练
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024),
                tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
            ])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)
```

**output**

```console
1 Physical GPU, 2 Logical GPUs
```

## Colab Demo

以下代码可以通过 Colab 链接测试效果 【 [TF2.3 Demo with Multiple GPUs](https://colab.research.google.com/drive/1tUffYwXf-KGXozjljUygzTTEgmCLtKgN?usp=sharing) 】

### Data Preparation

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

### Model Defination

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
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalAccuracy(), 
            metrics.SparseTopKCategoricalAccuracy(5)
        ]
    ) 
    return(model)

tf.keras.backend.clear_session()
```

### Model Training

```python
# 增加以下 2 行代码
strategy = tf.distribute.MirroredStrategy()  
with strategy.scope(): 
    model = create_model()
    model.summary()
    model = compile_model(model)

history = model.fit(ds_train, validation_data=ds_test, epochs=10)  
```

**output**

```console
WARNING:tensorflow:NCCL is not supported when using virtual GPUs, fallingback to reduction to one device
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1')
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
Epoch 1/10
WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/data/ops/multi_device_iterator_ops.py:601: get_next_as_optional (from tensorflow.python.data.ops.iterator_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Iterator.get_next_as_optional()` instead.
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:GPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:GPU:0', '/job:localhost/replica:0/task:0/device:GPU:1').
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
281/281 [==============================] - 7s 26ms/step - loss: 3.4208 - sparse_categorical_accuracy: 0.4520 - sparse_top_k_categorical_accuracy: 0.7200 - val_loss: 3.3410 - val_sparse_categorical_accuracy: 0.5263 - val_sparse_top_k_categorical_accuracy: 0.7182
Epoch 2/10
281/281 [==============================] - 6s 20ms/step - loss: 3.3315 - sparse_categorical_accuracy: 0.5343 - sparse_top_k_categorical_accuracy: 0.7270 - val_loss: 3.3304 - val_sparse_categorical_accuracy: 0.5343 - val_sparse_top_k_categorical_accuracy: 0.7208
Epoch 3/10
281/281 [==============================] - 6s 21ms/step - loss: 3.3171 - sparse_categorical_accuracy: 0.5488 - sparse_top_k_categorical_accuracy: 0.7256 - val_loss: 3.3268 - val_sparse_categorical_accuracy: 0.5378 - val_sparse_top_k_categorical_accuracy: 0.7204
Epoch 4/10
281/281 [==============================] - 6s 21ms/step - loss: 3.3116 - sparse_categorical_accuracy: 0.5538 - sparse_top_k_categorical_accuracy: 0.7241 - val_loss: 3.3268 - val_sparse_categorical_accuracy: 0.5378 - val_sparse_top_k_categorical_accuracy: 0.7150
Epoch 5/10
281/281 [==============================] - 6s 21ms/step - loss: 3.2967 - sparse_categorical_accuracy: 0.5714 - sparse_top_k_categorical_accuracy: 0.7251 - val_loss: 3.3268 - val_sparse_categorical_accuracy: 0.5405 - val_sparse_top_k_categorical_accuracy: 0.7155
Epoch 6/10
281/281 [==============================] - 6s 21ms/step - loss: 3.2738 - sparse_categorical_accuracy: 0.5942 - sparse_top_k_categorical_accuracy: 0.7275 - val_loss: 3.3084 - val_sparse_categorical_accuracy: 0.5561 - val_sparse_top_k_categorical_accuracy: 0.7195
Epoch 7/10
281/281 [==============================] - 6s 20ms/step - loss: 3.2635 - sparse_categorical_accuracy: 0.6034 - sparse_top_k_categorical_accuracy: 0.7269 - val_loss: 3.3122 - val_sparse_categorical_accuracy: 0.5521 - val_sparse_top_k_categorical_accuracy: 0.7217
Epoch 8/10
281/281 [==============================] - 6s 20ms/step - loss: 3.2582 - sparse_categorical_accuracy: 0.6077 - sparse_top_k_categorical_accuracy: 0.7279 - val_loss: 3.3010 - val_sparse_categorical_accuracy: 0.5663 - val_sparse_top_k_categorical_accuracy: 0.7177
Epoch 9/10
281/281 [==============================] - 6s 20ms/step - loss: 3.2554 - sparse_categorical_accuracy: 0.6098 - sparse_top_k_categorical_accuracy: 0.7270 - val_loss: 3.3011 - val_sparse_categorical_accuracy: 0.5659 - val_sparse_top_k_categorical_accuracy: 0.7208
Epoch 10/10
281/281 [==============================] - 6s 20ms/step - loss: 3.2545 - sparse_categorical_accuracy: 0.6104 - sparse_top_k_categorical_accuracy: 0.7266 - val_loss: 3.3071 - val_sparse_categorical_accuracy: 0.5583 - val_sparse_top_k_categorical_accuracy: 0.7208
```
