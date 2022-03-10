import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers

#标量模拟
out=tf.random.uniform([4,10])
print(out)
y=tf.constant([2, 3, 2, 0])
y=tf.one_hot(y, depth=10)
print(y)
#mean(squre(y_true-y_pred))
loss=tf.keras.losses.mse(y,out)
print(loss)

#向量模拟1
z=tf.random.normal([4,2])
b=tf.ones([2])
print(z)
z+=b
print(z)

#向量模拟2
fc=layers.Dense(3)
fc.build(input_shape=(2,4))
print(fc.bias)#维度为3

#矩阵模拟   模拟:x@w+b
x=tf.random.normal([2,4])
w=tf.ones([4,3])

b=tf.ones([3])
print(f'x@w+b:{x@w+b}')








print('end')