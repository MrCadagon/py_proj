import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers

#tensor索引
x=tf.random.normal([3, 4, 5, 6])
# x[0][0][0]=10   !error
print(x[0][0][1])
print(x[0, 0, 1])

#tensor切片
print(x[1:3])
print('test1')
print(x[:,::2,::2])
print('test2')

print(x[:, :, 1])

print()
print('test3')
print(x[...,1])

print()
print('test4')
print(x[1:, ...])



#tensor逆序
scalar1=tf.range(9)
print(scalar1[::-1])

