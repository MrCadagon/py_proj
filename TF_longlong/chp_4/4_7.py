import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers

#合理合并维度 reshape
x=tf.random.normal([2, 4, 4, 3])
x=tf.reshape(x,[2,-1])
x=tf.reshape(x,[2, 4, 4, 3])
# print(x)

#增删维度
x=tf.random.normal([3,3])
x=tf.expand_dims(x, axis=2)
x=tf.expand_dims(x, axis=0)
x=tf.expand_dims(x, axis=0)
x=tf.expand_dims(x, axis=0)
x=tf.squeeze(x, axis=0)
x=tf.squeeze(x)
print(x)

#交换维度
x=tf.random.normal([2, 3, 4, 5])
print('before')
print(x)
x=tf.transpose(x,perm=[3, 2, 1, 0])
print('efter')
print(x)

#数据复制
x=tf.constant([1,2])
x=tf.expand_dims(x, axis=0)
x=tf.tile(x, [2, 3])
print('AA')
print(x)

#broadcasting 右对齐扩展
z1=tf.constant([1, 2, 3, 4])
z1=tf.expand_dims(z1, axis=1)
z2=tf.constant([1, 2, 3])
z2=tf.expand_dims(z2, axis=0)
print(z1)
print(z2)
print(z1+z2)








print('end')
