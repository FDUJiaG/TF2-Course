# Modeling Procedure for Text

## Preparation of Data

Imdb 数据集的目标是根据电影评论的文本内容预测评论的情感标签

- 训练集有 20000 条电影评论文本
- 测试集有 5000 条电影评论文本
- 正面评论和负面评论都各占一半

文本数据预处理较为繁琐，包括 **中文切词**（本示例不涉及），**构建词典**，**编码转换**，**序列填充**，**构建数据管道** 等等

在 TensorFlow 中完成文本数据预处理的常用方案有两种

- 利用 `tf.keras.preprocessing` 中的 `Tokenizer` 词典构建工具和 `tf.keras.utils.Sequence` 构建文本数据生成器管道，较为复杂
- 使用 `tf.data.Dataset` 搭配 `tf.keras.layers.experimental.preprocessing.TextVectorization` 预处理层，为 TensorFlow 原生方式，相对也更加简单一些

以第二种方法为例，以下为数据集示例

![Imdb Dataset Eg](./figs/1-3-imdb-dataset-eg.jpg)

```python
import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import models, layers, preprocessing, optimizers, losses, metrics
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re,string

train_data_path = "./data/imdb/train.csv"
test_data_path =  "./data/imdb/test.csv"

MAX_WORDS = 10000   # 仅考虑最高频的 10000 个词
MAX_LEN = 200       # 每个样本保留 200 个词的长度
BATCH_SIZE = 20 


# 构建管道
def split_line(line):
    arr = tf.strings.split(line, "\t")
    label = tf.expand_dims(tf.cast(tf.strings.to_number(arr[0]), tf.int32), axis=0)
    text = tf.expand_dims(arr[1], axis=0)
    return (text, label)

ds_train_raw =  tf.data.TextLineDataset(filenames=[train_data_path]) \
   .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
   .shuffle(buffer_size=1000).batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)

ds_test_raw = tf.data.TextLineDataset(filenames=[test_data_path]) \
   .map(split_line, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
   .batch(BATCH_SIZE) \
   .prefetch(tf.data.experimental.AUTOTUNE)


# 构建词典
def clean_text(text):
    lowercase = tf.strings.lower(text)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    cleaned_punctuation = tf.strings.regex_replace(stripped_html,
         '[%s]' % re.escape(string.punctuation), '')
    return cleaned_punctuation

vectorize_layer = TextVectorization(
    standardize=clean_text,
    split='whitespace',
    max_tokens=MAX_WORDS-1,  # 有一个留给占位符
    output_mode='int',
    output_sequence_length=MAX_LEN)

ds_text = ds_train_raw.map(lambda text, label: text)
vectorize_layer.adapt(ds_text)
# print(vectorize_layer.get_vocabulary()[0:100])
for idx in range(11):
    print(vectorize_layer.get_vocabulary()[idx * 10:(idx + 1) * 10])


# 单词编码
ds_train = ds_train_raw.map(lambda text, label:(vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
ds_test = ds_test_raw.map(lambda text, label:(vectorize_layer(text), label)) \
    .prefetch(tf.data.experimental.AUTOTUNE)
```

**output**

```console
['', '[UNK]', 'the', 'and', 'a', 'of', 'to', 'is', 'in', 'it']
['i', 'this', 'that', 'was', 'as', 'for', 'with', 'movie', 'but', 'film']
['on', 'not', 'you', 'his', 'are', 'have', 'be', 'he', 'one', 'its']
['at', 'all', 'by', 'an', 'they', 'from', 'who', 'so', 'like', 'her']
['just', 'or', 'about', 'has', 'if', 'out', 'some', 'there', 'what', 'good']
['more', 'when', 'very', 'she', 'even', 'my', 'no', 'would', 'up', 'time']
['only', 'which', 'story', 'really', 'their', 'were', 'had', 'see', 'can', 'me']
['than', 'we', 'much', 'well', 'get', 'been', 'will', 'into', 'people', 'also']
['other', 'do', 'bad', 'because', 'great', 'first', 'how', 'him', 'most', 'dont']
['made', 'then', 'them', 'films', 'movies', 'way', 'make', 'could', 'too', 'any']
['after', 'characters', 'think', 'watch', 'many', 'two', 'seen', 'character', 'being', 'never']
```

## Define the Model

Keras 接口有以下 3 种方式构建模型

- 使用 Sequential 按层顺序构建模型
- 使用函数式 API 构建任意结构模型
- 继承 Model 基类构建自定义模型

此处选择使用继承 Model 基类构建自定义模型

```python
# 演示自定义模型范例，实际上应该优先使用 Sequential 或者函数式 API
tf.keras.backend.clear_session()

class CnnModel(models.Model):
    def __init__(self):
        super(CnnModel, self).__init__()
        
    def build(self,input_shape):
        self.embedding = layers.Embedding(MAX_WORDS, 7, input_length=MAX_LEN)
        self.conv_1 = layers.Conv1D(16, kernel_size= 5, name="conv_1", activation="relu")
        self.pool_1 = layers.MaxPool1D(name="pool_1")
        self.conv_2 = layers.Conv1D(128, kernel_size=2, name="conv_2", activation="relu")
        self.pool_2 = layers.MaxPool1D(name="pool_2")
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1, activation="sigmoid")
        super(CnnModel, self).build(input_shape)
    
    def call(self, x):
        x = self.embedding(x)
        x = self.conv_1(x)
        x = self.pool_1(x)
        x = self.conv_2(x)
        x = self.pool_2(x)
        x = self.flatten(x)
        x = self.dense(x)
        return(x)
    
    # 用于显示Output Shape
    def summary(self):
        x_input = layers.Input(shape=MAX_LEN, name="input")
        output = self.call(x_input)
        model = tf.keras.Model(inputs = x_input, outputs = output, name="model")
        model.summary()

model = CnnModel()
model.build(input_shape =(None, MAX_LEN))
model.summary()
```

**output**

```
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           [(None, 200)]             0         
_________________________________________________________________
embedding (Embedding)        (None, 200, 7)            70000     
_________________________________________________________________
conv_1 (Conv1D)              (None, 196, 16)           576       
_________________________________________________________________
pool_1 (MaxPooling1D)        (None, 98, 16)            0         
_________________________________________________________________
conv_2 (Conv1D)              (None, 97, 128)           4224      
_________________________________________________________________
pool_2 (MaxPooling1D)        (None, 48, 128)           0         
_________________________________________________________________
flatten (Flatten)            (None, 6144)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 6145      
=================================================================
Total params: 80,945
Trainable params: 80,945
Non-trainable params: 0
_________________________________________________________________
```

## Training Model

训练模型通常有 3 种方法

- 内置 `fit` 方法
- 内置 `train_on_batch` 方法
- 自定义训练循环

此处通过自定义训练循环训练模型

```python
# 打印时间分割线
@tf.function
def printbar():
    today_ts = tf.timestamp() % (24 * 60 * 60)
    
    hour = tf.cast(today_ts // 3600 + 8, tf.int32) % tf.constant(24)
    minite = tf.cast((today_ts % 3600) // 60, tf.int32)
    second = tf.cast(tf.floor(today_ts % 60), tf.int32)
    
    def timeformat(m):
        if tf.strings.length(tf.strings.format("{}",m)) == 1:
            return(tf.strings.format("0{}", m))
        else:
            return(tf.strings.format("{}", m))

    timestring = tf.strings.join([timeformat(hour), timeformat(minite),
                timeformat(second)], separator=":")
    tf.print("==========" * 9 + timestring)
```

自定义训练过程

```python
optimizer = optimizers.Nadam()
loss_func = losses.BinaryCrossentropy()

train_loss = metrics.Mean(name='train_loss')
train_metric = metrics.BinaryAccuracy(name='train_accuracy')

valid_loss = metrics.Mean(name='valid_loss')
valid_metric = metrics.BinaryAccuracy(name='valid_accuracy')


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
    predictions = model(features, training=False)
    batch_loss = loss_func(labels, predictions)
    valid_loss.update_state(batch_loss)
    valid_metric.update_state(labels, predictions)


def train_model(model, ds_train, ds_valid, epochs):
    for epoch in tf.range(1, epochs + 1):
        
        for features, labels in ds_train:
            train_step(model, features, labels)

        for features, labels in ds_valid:
            valid_step(model, features, labels)
        
        # 此处 logs 模板需要根据 metric 具体情况修改
        logs = 'Epoch={}, Loss:{}, Accuracy:{}, Valid Loss:{}, Valid Accuracy:{}' 
        
        if epoch % 1 == 0:
            printbar()
            tf.print(
                tf.strings.format(
                    logs,
                    (
                        epoch, 
                        train_loss.result(),
                        train_metric.result(),
                        valid_loss.result(),
                        valid_metric.result()
                    )
                ))
            tf.print("")
        
        train_loss.reset_states()
        valid_loss.reset_states()
        train_metric.reset_states()
        valid_metric.reset_states()

train_model(model, ds_train, ds_test, epochs = 6)
```

**output**

```console
==========================================================================================11:47:47
Epoch=1, Loss:0.170583129, Accuracy:0.93625, Valid Loss:0.361590534, Valid Accuracy:0.8712

==========================================================================================11:47:58
Epoch=2, Loss:0.113764673, Accuracy:0.95955, Valid Loss:0.439444721, Valid Accuracy:0.8708

==========================================================================================11:48:09
Epoch=3, Loss:0.0647901, Accuracy:0.9784, Valid Loss:0.602329791, Valid Accuracy:0.8648

==========================================================================================11:48:19
Epoch=4, Loss:0.0350493118, Accuracy:0.9885, Valid Loss:0.800360799, Valid Accuracy:0.86

==========================================================================================11:48:28
Epoch=5, Loss:0.0180068184, Accuracy:0.9945, Valid Loss:0.981879413, Valid Accuracy:0.859

==========================================================================================11:48:38
Epoch=6, Loss:0.0143510057, Accuracy:0.9955, Valid Loss:1.15362406, Valid Accuracy:0.854
```

## Evaluation Model

通过自定义训练循环训练的模型没有经过编译，无法直接使用 `model.evaluate(ds_valid)` 方法

```python
def evaluate_model(model, ds_valid):
    for features, labels in ds_valid:
         valid_step(model, features,labels)
    logs = 'Valid Loss: {}, Valid Accuracy: {}' 
    tf.print(tf.strings.format(logs, (valid_loss.result(), valid_metric.result())))
    
    valid_loss.reset_states()
    train_metric.reset_states()
    valid_metric.reset_states()
```

执行自定义的训练模型

```python
evaluate_model(model, ds_test)
```

**output**

```console
Valid Loss: 1.15362406, Valid Accuracy: 0.854
```

## Use the Model

可以使用以下方法

* `model.predict(ds_test)`
* `model(x_test)`
* `model.call(x_test)`
* `model.predict_on_batch(x_test)`

推荐优先使用 `model.predict(ds_test)` 方法，既可以对 Dataset，也可以对 Tensor 使用

```python
model.predict(ds_test)
```

**output**

```console
array([[0.6412138 ],
       [1.        ],
       [1.        ],
       ...,
       [0.996657  ],
       [0.99167955],
       [1.        ]], dtype=float32)
```

其他方法示例

```python
for x_test, _ in ds_test.take(1):
    print(model(x_test))
    # 以下方法等价
    # print(model.call(x_test))
    # print(model.predict_on_batch(x_test))
```

**output**

```console
tf.Tensor(
[[6.4121389e-01]
 [1.0000000e+00]
 [1.0000000e+00]
 [1.6712733e-08]
 [9.9923491e-01]
 [1.7198188e-08]
 [1.0513377e-09]
 [2.0263493e-03]
 [9.9999881e-01]
 [9.9999964e-01]
 [1.0000000e+00]
 [9.9939692e-01]
 [3.8780005e-08]
 [9.9943322e-01]
 [2.7051968e-07]
 [9.7854507e-01]
 [1.8463733e-07]
 [6.7190182e-01]
 [8.2147717e-03]
 [9.9381411e-01]], shape=(20, 1), dtype=float32)
```

## Save the Model

推荐使用 TensorFlow 原生方式保存模型

```python
model.save('./data/tf_model_savedmodel', save_format="tf")

model_loaded = tf.keras.models.load_model('./data/tf_model_savedmodel')
model_loaded.predict(ds_test)
```

**output**

```console
INFO:tensorflow:Assets written to: ./data/tf_model_savedmodel/assets

array([[0.6412138 ],
       [1.        ],
       [1.        ],
       ...,
       [0.996657  ],
       [0.99167955],
       [1.        ]], dtype=float32)
```
