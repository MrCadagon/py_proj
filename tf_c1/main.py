

# 输出字符串
import tensorflow as tf
from tensorflow import keras

# the first code
import tensorflow as tf
A = tf.constant([[1, 2], [3, 4]])
B = tf.constant([[5, 6], [7, 8]])
C = tf.matmul(A, B)
print(C)



# tf.compat.v1.disable_eager_execution()
# hello=tf.constant('Hello,Tensorflow!')
# sess=tf.compat.v1.Session()
# print(sess.run(hello))

# a=tf.constant(2.)
# b=tf.constant(4.)
# print('a+b=',a+b)




# a=1.2
# aa=tf.constant([1.2,1.3])
# print(type(a),type(aa),tf.is_tensor(aa))
# print(aa)





# a=tf.constant([1.,2.],name="a")
# tf.test.gpu_device_name()
# print(a.graph is tf.get_default_graph())




# import numpy as np
# X_raw=np.array([2013,2014,2015,2016,2017],dtype=np.float32)
# Y_ray=np.array([12000, 14000, 15000, 16500, 17500],dtype=np.float32)
# X=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
# Y=(Y_ray-Y_ray.min())/(Y_ray.max()-Y_ray.min())
#
# X=tf.constant(X)
# Y=tf.constant(Y)
#
# a=tf.Variable(initial_value=0.)
# b=tf.Variable(initial_value=0.)
# variable=[a,b]
#
# loop=10000
# optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3)
# for i in range(loop):
#     with tf.GradientTape() as tape:
#         y_pred=a*X+b
#         loos=0.5*tf.reduce_sum(tf.square(y_pred-Y))
#     grads=tape.gradient(loos,variable)
#     #利用梯度和学习率对 ab进行更新
#     optimizer.apply_gradients(grads_and_vars=zip(grads,variable))
# print(a,b)

import tensorflow as tf

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


# 以下代码结构与前节类似
model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)

