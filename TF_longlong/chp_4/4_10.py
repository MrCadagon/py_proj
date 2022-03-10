import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets

# 显示GPU运行情况
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
tf.debugging.set_log_device_placement(True)

# mnist 前向传播
(x, y), _ = datasets.mnist.load_data()
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.int32)
y = tf.one_hot(y, depth=10)

print(x.shape)
print(x.dtype)
print(y.shape)
print(y.dtype)

# 创建迭代器 获取前128个数据
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
train_iter = next(train_iter)
for a in train_iter:
    print(a.shape)

# /词典无法查看shape  用iter方便迭代
# test={'1':'a',
#       '2':'b',
#       '3':'c'}
# print(test.shape)
# 词典无法查看shape  用iter方便迭代/

# 创建参数wb
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([128]))
b3 = tf.Variable(tf.zeros([10]))

lr = 1e-3
for epoch in range(2000):
    tolal_match = tf.constant(0)
    tolal_num = tf.constant(0)
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 28 * 28])
        with tf.GradientTape() as tape:
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            h3 = tf.nn.relu(h2 @ w3 + b3)

            loss = tf.square(y - h3)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        # #show the process
        # if step % 100 == 0:
        #     print(epoch, step, 'losses', loss)

        # calcu the right ratio
        pred = tf.argmax(h3, axis=1)
        ideal = tf.argmax(y, axis=1)
        compare = tf.equal(pred, ideal)
        compare = tf.cast(compare, dtype=tf.int32)
        tolal_match += tf.reduce_sum(compare)
        tolal_num += 128

    print('the No [', epoch, '] match ratio is :',
          tf.cast(tolal_match, dtype=tf.float32) / tf.cast(tolal_num, dtype=tf.float32))

print('end')
