# Chapter 05: High-level API in TensorFlow

TensorFlow 的高阶 API 主要是 `tensorflow.keras.models`，主要包括

- **模型的构建**，Sequential、Functional API，Model 子类化
- **模型的训练**，内置 `fit` 方法，内置 `train_on_batch` 方法，自定义训练循环，单 GPU 训练模型，多 GPU 训练模型，TPU 训练模型
- **模型的部署**，TensorFlow Serving 部署模型，使用 Spark（Scala）调用 TensorFlow 模型
