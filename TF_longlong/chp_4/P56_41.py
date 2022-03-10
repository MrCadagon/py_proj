#1.标量 向量
import tensorflow as tf
import numpy as np

a=1.2
aa=tf.constant(1.2)

print(f'type(a):{type(a)}  type(aa):{type(aa)}')
print(f'tf.is_tensor(aa):{tf.is_tensor(aa)}')

x=tf.constant([1,2,3])
print(x)
print(x.numpy(), x.shape)

#2. 字符串
print()
str=tf.constant('Hello,Deep Learning')
print(str, tf.strings.lower(str))
tf.strings.lower(str)
tf.strings.length(str)
str2=tf.strings.split(str,',')
print(f'str2:{str2}')
for ss in str2:
    print(ss)

#3. 布尔类型
bo=tf.constant([True, False])




