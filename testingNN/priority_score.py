#!/usr/bin/env python
# coding: utf-8


import os
# # GPU Settings
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from keras.datasets import mnist
from keras.models import load_model
from keras.utils import np_utils


img_rows, img_cols = 28, 28
num_classes = 10

# ### Reshape the Data

# Reshape to be samples * pixels * width * height
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(
    x_train.shape[0], img_rows, img_cols, 1
).astype('float32')
x_test = x_test.reshape(
    x_test.shape[0], img_rows, img_cols, 1
).astype('float32')

# ### Data Normalization

# Normalize inputs from 0-255 to 0-1
x_train = x_train / 255
x_test = x_test / 255

# ### One-Hot Encode

y_train = np_utils.to_categorical(y_train, num_classes)  # Converts a class vector (integers) to binary class matrix.
y_test = np_utils.to_categorical(y_test, num_classes)


# ## Priority Score
# ### Reload the Model

model_dir = os.path.join(os.path.abspath('.'), "models")
model_name = "lenet1.h5"
model_path = os.path.join(model_dir, model_name)

if not os.path.exists(model_dir):
    print("[INFO]", model_dir, "Directory does NOT Existed")
    sys.exit(1)

elif not os.path.exists(model_path):
    print("[INFO] No", model_name, "Model File in the Path", model_dir)
    sys.exit(1)

load_model = load_model(model_path)
print("[INFO]", model_name, "Model File is Successfully Loaded!")

load_model.summary()

print("\n[Model Evaluation]")
evaluate = load_model.evaluate(x_test, y_test, verbose=1)

# ### Calculate the Priority Score

def xi_score(idx, model, base):
    x_input = x_test[idx].reshape(-1, img_rows, img_cols, 1)
#     predict_label = model.predict_classes(x_input)[0]
    predict_list = model.predict(x_input)
    predict_label = np.argmax(predict_list, axis=-1)[0]

    x_score = - np.inner(
        predict_list[0],
        np.divide(np.log(predict_list)[0], np.log(base))
    )

    return predict_list[0], predict_label, x_score

# ### Define a Print Function

def xi_print(p_lst, p_class, r_class, score):

    col_name = ["Label", "Predict_Probability", "FCT"]
    col_len = [len(item) for item in col_name]
    total_len = sum(col_len) + 3 * len(col_len) - 1

    title = "Predict Table"

    print(title.center(len(title) + 2).center(total_len, "="))
    print('_' * total_len)
    print(
        "", col_name[0], 
        "|", col_name[1], 
        "|", col_name[2], ""
    )
    print(
        "", "-" * col_len[0], 
        "|", "-" * col_len[1], 
        "|", "-" * col_len[2], ""
    )

    pred_one_hot = np_utils.to_categorical(p_class, num_classes)

    for idx, item in zip(range(len(p_lst)), p_lst):
        print(
            "", str(idx).center(col_len[0]), "|", 
            "%.6e".rjust(col_len[1] - 8) % item, "|", 
            str("*" * int(pred_one_hot[idx])).center(col_len[2]), ""
        )

    print('=' * total_len)
    print(" Realistic Class :", r_class)
    print(" Predict Class   :", p_class)
    print(" Predict Validity:", p_class == r_class)
    print(" Priority Score  :", score)
    print('_' * total_len)

    return True

# ### Example

# Single Example
index = np.random.choice(range(y_test.shape[0]))
print("\n[Single Example] Index:", index)
p_list, p_label, x_score = xi_score(index, load_model, num_classes)
output = xi_print(p_list, p_label, np.argmax(y_test[index]), x_score)

# More Examples
key_in = ""
seed = 0
quit_input = ["y", "yes"]
while key_in.lower() not in quit_input:
    seed += 1
    # np.random.seed(seed)
    index = np.random.choice(range(y_test.shape[0]))
    print(
        "\n[Eg." + str(seed).rjust(3, "0") + \
        "] Index." + str(index).rjust(4, "0") + ":"
    )

    p_list, p_label, x_score = xi_score(index, load_model, num_classes)
    output = xi_print(p_list, p_label, np.argmax(y_test[index]), x_score)

    im = plt.imshow(x_test[index].reshape(img_rows, img_cols), cmap='gray')
    plt.show()

    if output:
        time.sleep(0.5)
        key_in = input("\nQuit for the Example? yes/[No]: ")
    else:
        break
