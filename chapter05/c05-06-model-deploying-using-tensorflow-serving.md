# 05-06 Model Deploying Using Tensorflow-Serving

TensorFlow 训练好的模型以 TensorFlow 原生方式保存成 **protobuf 文件** 后可以用许多方式部署运行，例如

- 通过 **tensorflow-js** 可以用javascrip脚本加载模型并在浏览器中运行模型
- 通过 **tensorflow-lite** 可以在移动和嵌入式设备上加载并运行 TensorFlow 模型
- 通过 **tensorflow-serving** 可以加载模型后提供网络接口 API 服务，通过任意编程语言发送网络请求都可以获取模型预测结果
- 通过 **tensorFlow for Java** 接口，可以在 Java 或者 spark(scala) 中调用 TensorFlow 模型进行预测

这里主要介绍 **tensorflow serving** 部署模型，下节介绍使用 **spark(scala)** 调用 TensorFlow 模型的方法

## Overview of Tensorflow Serving Model Deployment

使用 tensorflow serving 部署模型要完成以下步骤

1. 准备 protobuf 模型文件
1. 安装 tensorflow serving
1. 启动 tensorflow serving 服务
1. 向 API 服务发送请求，获取预测结果

```python
%tensorflow_version 2.x
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras import *
```

## Preparing the Protobuf Model File

使用 `tf.keras` 训练一个简单的线性回归模型，并保存成 `protobuf` 文件

```python
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

## 样本数量
n = 800

## 生成测试用数据集
X = tf.random.uniform([n, 2], minval=-10, maxval=10) 
w0 = tf.constant([[2.0], [-1.0]])
b0 = tf.constant(3.0)

Y = X @ w0 + b0 + tf.random.normal(
    [n, 1], mean=0.0, stddev=2.0)   # @ 表示矩阵乘法，增加正态扰动

## 建立模型
tf.keras.backend.clear_session()
inputs = layers.Input(shape=(2,), name="inputs")    # 设置输入名字为 inputs
outputs = layers.Dense(1, name="outputs")(inputs)   # 设置输出名字为 outputs
linear = models.Model(inputs=inputs, outputs=outputs)
linear.summary()

## 使用 fit 方法进行训练
linear.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
linear.fit(X, Y, batch_size=8, epochs=100)

tf.print("w =", linear.layers[1].kernel)
tf.print("b =", linear.layers[1].bias)

## 将模型保存成 pb 格式文件
export_path = "./data/linear_model/"
version = "1"       # 后续可以通过版本号进行模型版本迭代与管理
linear.save(export_path + version, save_format="tf") 
```

查看保存的模型文件，`!` 后可执行 shell 命令，仅支持 Jupyter 或者 IPython，纯的 Python 命令行不支持

```jupyter
!ls {export_path+version}
```

**output**

```console
assets	saved_model.pb	variables
```

查看模型文件相关信息

```jupyter
!saved_model_cli show --dir {export_path+str(version)} --all
```

**output**

```console
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['__saved_model_init_op']:
  The given SavedModel SignatureDef contains the following input(s):
  The given SavedModel SignatureDef contains the following output(s):
    outputs['__saved_model_init_op'] tensor_info:
        dtype: DT_INVALID
        shape: unknown_rank
        name: NoOp
  Method name is: 

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['inputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2)
        name: serving_default_inputs:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['outputs'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: StatefulPartitionedCall:0
  Method name is: tensorflow/serving/predict
WARNING:tensorflow:From /tensorflow-2.1.0/python3.6/tensorflow_core/python/ops/resource_variable_ops.py:1786: calling BaseResourceVariable.__init__ (from tensorflow.python.ops.resource_variable_ops) with constraint is deprecated and will be removed in a future version.
Instructions for updating:
If using Keras pass *_constraint arguments to layers.

Defined Functions:
  Function Name: '__call__'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None

  Function Name: '_default_save_signature'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')

  Function Name: 'call_and_return_all_conditional_losses'
    Option #1
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: True
        Argument #3
          DType: NoneType
          Value: None
    Option #2
      Callable with:
        Argument #1
          inputs: TensorSpec(shape=(None, 2), dtype=tf.float32, name='inputs')
        Argument #2
          DType: bool
          Value: False
        Argument #3
          DType: NoneType
          Value: None
```

## Installing tensorflow serving

安装 **tensorflow serving** 有 2 种主要方法

- 通过 Docker 镜像安装，是最简单，最直接的方法，**推荐** 采用
- 通过 `apt` 安装

Docker 可以理解成一种容器，其上面可以给各种不同的程序提供独立的运行环境，一般业务中用到 TensorFlow 的企业都会有运维同学通过 Docker 搭建 tensorflow serving，无需算法工程师同学动手安装，以下安装过程仅供参考

不同操作系统机器上安装 Docker 的方法可以参照以下链接

- [Windows](https://www.runoob.com/docker/windows-docker-install.html)
- [MacOS](https://www.runoob.com/docker/macos-docker-install.html)
- [CentOS](https://www.runoob.com/docker/centos-docker-install.html)

安装 Docker 成功后，使用如下命令加载 `tensorflow/serving` 镜像到 Docker 中

```bash
docker pull tensorflow/serving
```

如果遇到一些 `i/o timeout` 的问题，可以参考这篇博客 【[ Docker 及服务器遇到的坑](https://www.cnblogs.com/followyou/p/10315717.html) 】

## Starting the Tensorflow Serving

```jupyter
!docker run -t --rm -p 8501:8501 \
    -v "/Users/.../data/linear_model/" \
    -e MODEL_NAME=linear_model \
    tensorflow/serving & >server.log 2>&1
```

## Sending Requests to API Services

可以使用任何编程语言的 `http` 功能发送请求，下面示范 Linux 的 `curl` 命令发送请求，以及 Python 的 `requests` 库发送请求

```jupyter
!curl -d '{"instances": [[1.0, 2.0], [5.0,7.0]]}' \
    -X POST http://localhost:8501/v1/models/linear_model:predict
```

**output**

```console
{
    "predictions": [[3.06546211], [6.02843142]]
}
```

```python
import json, requests

data = json.dumps({"signature_name": "serving_default", "instances": [[1.0, 2.0], [5.0, 7.0]]})
headers = {"content-type": "application/json"}
json_response = requests.post('http://localhost:8501/v1/models/linear_model:predict', 
        data=data, headers=headers)
predictions = json.loads(json_response.text)["predictions"]
print(predictions)
```

**output**

```console
[[3.06546211], [6.02843142]]
```
