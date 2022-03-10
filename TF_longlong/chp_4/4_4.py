#4.4 创建张量
import tensorflow as tf
import numpy as np
x1=tf.convert_to_tensor([1, 2.])
x2=tf.constant([1, 2.])
x2=tf.constant(np.array([1, 2.]))

#创建全0 1张量
x3=tf.ones([3, 2])
x3=tf.zeros([3, 2])
x4=tf.zeros_like(x3)
x4=tf.ones_like(x3)
print(x3)
print(x4)

#填充fill函数
x5=tf.fill([],-1)
x5=tf.fill([1],-1)
x5=tf.fill([3, 3],-1)

#创建已知分布的张量  属于random变量
x1=tf.random.normal([2, 2], mean=1, stddev=2)
x2=tf.random.normal([2, 2])
x3=tf.random.uniform([2, 2], maxval=100, dtype=tf.float64)

#创建序列
x1=tf.range(1, 10, delta=2)


print('end')