import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x = tf.random.uniform([4, 35, 8], dtype=tf.float64)
x1 = tf.gather(x, [1, 3, 5, 7], axis=1)
# print(x1)

x = tf.random.uniform([4, 35, 8], dtype=tf.float64)
x1 = tf.stack([x[1, 1], x[2, 2], x[3, 3]], axis=0)
x1 = tf.gather(x, [1, 2, 3], axis=0)

x = tf.random.uniform([4, 35, 8], dtype=tf.float64)
# 获取多个目标张量 上式推广
x1 = tf.gather_nd(x, [[1, 1], [2, 2], [3, 3]])
x1 = tf.gather_nd(x, [[1, 1, 1], [2, 2, 2], [3, 3, 3]])

# 使用掩码方式采样
mask1 = tf.constant([1, 0, 0, 1])
mask1 = tf.cast(mask1, tf.bool)
x1 = tf.boolean_mask(x, mask1, axis=0)
# 使用掩码方式采样 2
x = tf.random.uniform([2, 3, 8])
mask1 = tf.constant([[1, 1, 0], [1, 0, 0]])
mask1 = tf.cast(mask1, dtype=tf.bool)
x1 = tf.boolean_mask(x, mask1)

# 按规则选取
a = tf.range(1, 10)
a = tf.reshape(a, [3, 3])
b = a * -1
mask_where = tf.constant([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
mask_where = tf.cast(mask_where, tf.bool)
result = tf.where(mask_where, a, b)

# 获取大于0的元素
x = tf.random.normal([3, 3, 3])
mask = x > 0
indexs = tf.where(mask)
pos_ele = tf.gather_nd(x, indexs)
# or
pos_ele = tf.boolean_mask(x, mask)

# 赋值
index_axis0 = tf.constant([[1], [3]])
data_scatter = tf.constant([
    [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]],
    [[5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8], [5, 6, 7, 8]]
])
result = tf.scatter_nd(index_axis0, data_scatter, [4, 4, 4])

for x in range(-8, 8, 100):
    print(x)

#work---绘制xy网格图形
x=tf.linspace(-8.,8,100)
y=tf.linspace(-8.,8,100)
x,y=tf.meshgrid(x,y)
z=tf.math.sin(x**2+y**2)/(x**2+y**2)

fig=plt.figure()
ax=Axes3D(fig)
ax.contour(x.numpy(), y.numpy(), z.numpy(), 20)
plt.show()




print('end')
