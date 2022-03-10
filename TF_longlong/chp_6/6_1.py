import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras import Sequential

# 层方式实现全连接层
x = tf.random.normal([4, 28 * 28])
fc = layers.Dense(512, activation=None)
h1 = fc(x)
print(fc.kernel)
print(fc.bias)
print(fc.trainable_variables)
print(fc.variables)

# 神经网络层方式实现
model1 = tf.keras.Sequential([
    layers.Dense(512, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(10, activation=None)
])
out = model1(x)

# 激活函数
x = tf.linspace(-5., 5., 10)
out1 = tf.sigmoid(x)#NO2
out2 = tf.nn.relu(x)
out3 = tf.nn.leaky_relu(x, alpha=0.1)
out4 = tf.nn.tanh(x)
out5=tf.nn.softmax(x)#NO3

print('end')
