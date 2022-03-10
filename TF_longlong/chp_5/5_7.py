import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

(x, y), (x_test, y_test) = datasets.mnist.load_data()
train_db = tf.data.Dataset.from_tensor_slices((x, y))
# 随机打散 批训练
db = train_db.shuffle(10000)  # 对前10000个元素shuffle
db = db.batch(128)


# 预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [-1, 28 * 28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return (x, y)


db = db.map(preprocess)

print('end')
