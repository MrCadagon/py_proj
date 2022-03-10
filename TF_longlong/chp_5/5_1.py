import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 拼接 堆叠
a = tf.random.normal([4, 35, 8])
b = tf.random.normal([6, 35, 8])
a = tf.concat([a, b], axis=0)

a = tf.random.normal([35, 8])
b = tf.random.normal([35, 8])
a = tf.stack([a, b], axis=0)

# 分割
a = tf.random.normal([10, 35, 8])
result = tf.unstack(a, axis=0)
result = tf.split(a, axis=0, num_or_size_splits=10)
result = tf.split(a, axis=0, num_or_size_splits=[2, 4, 2, 2])

# 范数
a = tf.random.normal([10, 35, 8])
L1 = tf.norm(a, ord=1)
L2 = tf.norm(a, ord=2)
L_inf = tf.norm(a, ord=np.inf)

# 最大最小值 均值 和
x = tf.random.normal([4, 10])
print(tf.reduce_min(x, axis=1))
print(tf.reduce_max(x, axis=-1))
print(tf.reduce_sum(x, axis=1))
print(tf.reduce_mean(x, axis=-1))

print(tf.reduce_min(x))
print(tf.reduce_max(x))
print(tf.reduce_sum(x))
print(tf.reduce_mean(x))

#最值在某一维度的索引
print(tf.nn.softmax(x, axis=1))
print(tf.argmax(x, axis=1))
print(tf.argmin(x, axis=1))



print('end')
