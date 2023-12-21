import numpy as np
import tensorflow as tf
from keras.datasets import mnist
# from tensorflow.keras.datasets import mnist

# Prepare MNIST data.
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Convert to float32.
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# Normalize images value from [0, 255] to [0, 1].
# 训练数据是8位灰度图像，取值为［0,255］，因此在输入时需要将其归一化到［0,1］区间，以利于模型训练
x_train, x_test = x_train / 255., x_test / 255.
# Build an RNN
# 这里的样例代码使用了隐藏层维度为80的单层LSTM，然后通过全连接层将其输出变换为10维，用于分类
# model = tf.keras.Sequential([tf.keras.layers.LSTM(80),
model = tf.keras.Sequential([tf.keras.layers.SimpleRNN(80),
                             tf.keras.layers.Dense(10)])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32)
model.evaluate(x_test, y_test, batch_size=64)
