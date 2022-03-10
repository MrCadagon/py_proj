#数值精度
import tensorflow as tf
import numpy as np

print(tf.constant(12222222,dtype=tf.int32))
print(tf.constant(np.pi, dtype=tf.float64))

#cast 类型转换  tf.double
a=tf.constant(3.122, dtype=tf.float16)
print('before:', a.dtype)
if a.dtype!=tf.float64:
    a=tf.cast(a,tf.double)
print('after:', a.dtype)

#4.3 待优化张量
va=tf.constant(1.22, dtype=tf.float64)
vaa=tf.Variable(va)
print(vaa.name)
print(vaa.trainable)



















