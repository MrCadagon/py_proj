import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 张量比较 测试test
sample = tf.random.uniform([100, 10])
sample = tf.argmax(sample, axis=1)
label = tf.random.uniform([100], maxval=10, dtype=tf.int64)  # 注意单位 int64  int32不可

result = tf.equal(sample, label)
result = tf.cast(result, dtype=tf.int64)
match_ratio = tf.cast(tf.reduce_sum(result), dtype=tf.float64) / 100.

# 填充 复制
# 每个维度都填充
x = tf.random.uniform([4, 28, 28, 1])
x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])
x = tf.tile(x, [1, 2, 2, 3])

# 限幅
x = range(10)
x = tf.maximum(x, 7)
x = range(10)
x = tf.minimum(x, 7)
x = range(10)
x = tf.clip_by_value(x, 2, 7)



print('end')
