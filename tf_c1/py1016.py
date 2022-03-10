#自动求导001

# import tensorflow as tf
# x=tf.Variable(initial_value=3.)
# with tf.GradientTape() as tape:
#     y=tf.square(x)
# y_grad=tape.gradient(y,x);
# print(y,y_grad)

#自动求导002
# import tensorflow as tf
# x=tf.constant([[1.,2.],[3.,4.]])
# y=tf.constant([[1.],[2.]])
# w=tf.Variable(initial_value=[[1.],[2.]])
# b=tf.Variable(initial_value=1.)
# with tf.GradientTape() as tape:
#     # x * w
#     L=0.5*tf.reduce_sum(tf.square(tf.matmul(x,w)+b-y))
# L_grad_w,L_grad_b=tape.gradient(L,[w,b])
# print(L,L_grad_w,L_grad_b)

#预测房价  基础的梯度下降
# import numpy as np
# X_raw=np.array([2013,2014,2015,2016,2017],dtype=np.float32)
# Y_raw=np.array([12000, 14000, 15000, 16500, 17500],dtype=np.float32)
# X=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
# Y=(Y_raw-Y_raw.min())/(Y_raw.max()-Y_raw.min())
#
# a,b=0,0
# num_loop=10000
# le=5e-4
# for e in range(num_loop):
#     Y_pred=a*X-b
#     grad_a,grad_b=2.*(Y_pred-Y).dot(X),2.*(Y_pred-Y).sum()
#     a,b=a-grad_a*le,b-grad_b*le
# print(a,b)

#预测房价  tf线性回归
import tensorflow as tf
import numpy as np
# X_raw=np.array([2013,2014,2015,2016,2017],dtype=np.float32)
# Y_raw=np.array([12000, 14000, 15000, 16500, 17500],dtype=np.float32)
# X=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
# Y=(Y_raw-Y_raw.min())/(Y_raw.max()-Y_raw.min())
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
# for e in range(loop):
#     with tf.GradientTape() as tape:
#         Y_pred = a * X + b
#         loss = 0.5 * tf.reduce_sum(tf.square(Y_pred - Y))
#     grads=tape.gradient(loss,variable)
#     optimizer.apply_gradients(grads_and_vars=zip(grads,variable))
# print(a,b)


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


# P89 book
# tf.compat.v1.disable_eager_execution()
# a=tf.constant([1.0,2.0],name="a")
# # print(a.graph is tf.get_default_graph() )
# print(a.graph is tf.compat.v1.get_default_graph())

# 上一章中简单的线性模型 y_pred = a * X + b ，我们可以通过模型类的方式编写如下：(part6)

import tensorflow as tf
X=tf.constant([[1.,2.,3.],[4.,5.,6.]])
Y=tf.constant([[10.],[20.]])

class my_linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense=tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer
        )
    def call(self,input):
        output=self.dense(input)
        return output












