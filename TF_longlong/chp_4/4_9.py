import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers

#数学运算 整除 整余 乘方 指数 对数 logxy
a=tf.constant([1, 2, 3, 4, 5])
print(a)
b=tf.constant(2)
print(a//b)
print(a%b)

a=tf.cast(a, dtype=tf.float64)
print(a**(2))
print(tf.square(a))
print(a**(0.5))
print(tf.sqrt(a))

print(tf.exp(2.))
print(tf.math.log(tf.exp(2.)))
#实现logxy
x=tf.constant(3.)
y=tf.constant(2.)
print(tf.math.log(y)/tf.math.log(x))

#矩阵相乘  broading casting
a=tf.random.normal([4, 6, 32])
b=tf.random.normal([1, 32, 2])  #ok
b=tf.random.normal([32, 2])  #ok
print(a@b)