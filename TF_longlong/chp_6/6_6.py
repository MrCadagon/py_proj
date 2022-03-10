import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras import Sequential

#计算误差
#MSE
o = tf.random.normal([2, 10])
y = tf.constant([1, 3])
y = tf.one_hot(y, depth=10)
mse = tf.losses.MSE(y, o)
#交叉熵




print('end')