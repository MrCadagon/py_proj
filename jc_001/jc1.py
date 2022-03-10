# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print("Hi, {0}".format(name))  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import tensorflow as tf

# 1
# random_float=tf.random.uniform(shape=())
#
# random_vector=tf.zeros(shape=(2))
#
# print(random_float,random_vector)
#
# A=tf.constant([[1.,2.],[3.,4.]])
# B=tf.constant([[1.,2.],[3.,4.]])
# C=tf.matmul(A,B);
# print(C)

# 2
# x=tf.Variable(initial_value=3.)
# with tf.GradientTape() as tape:
#     y=tf.square(x)
# y_grad=tape.gradient(y,x)
# print(y,y_grad)

#2.2
# x=tf.constant([[1.,2.],[3.,4.]])
# y=tf.constant([[1.],[2.]])
# w=tf.Variable(initial_value=[[1.],[2.]])
# b=tf.Variable(initial_value=1.)
# with tf.GradientTape() as tape:
#     L=0.5*tf.reduce_sum(tf.square(tf.matmul(x,w)+b-y))
# w_grad,b_grad=tape.gradient(L,[w,b])
# # L,w_grad,b_grad
# print('the L is ',L)
# print('the w_grad is ',w_grad)
# print('the b_grad is ',b_grad)


#3.1  利用numpy手动求线性回归

# import numpy as np
# X_raw=np.array([2013,2014,2015,2016,2017],dtype=np.float32)
# Y_ray=np.array([12000, 14000, 15000, 16500, 17500],dtype=np.float32)
# X=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
# Y=(Y_ray-Y_ray.min())/(Y_ray.max()-Y_ray.min())
#
# a,b=0,0
# num_epoch=10000
# lr=5e-4
# for e in range(num_epoch):
#     y_pred=a*X+b
#     grad_a,grad_b=2.*(y_pred-Y).dot(X),2.*(y_pred-Y).sum()
#     a,b=a-lr*grad_a,b-lr*grad_b
#
# print(a,b)

#3.2  利用tf手动求线性回归
import numpy as np
X_raw=np.array([2013,2014,2015,2016,2017],dtype=np.float32)
Y_ray=np.array([12000, 14000, 15000, 16500, 17500],dtype=np.float32)
X=(X_raw-X_raw.min())/(X_raw.max()-X_raw.min())
Y=(Y_ray-Y_ray.min())/(Y_ray.max()-Y_ray.min())

X=tf.constant(X)
Y=tf.constant(Y)

a=tf.Variable(initial_value=0.)
b=tf.Variable(initial_value=0.)
variable=[a,b]

loop=10000
optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3)
for i in range(loop):
    with tf.GradientTape() as tape:
        y_pred=a*X+b
        loos=0.5*tf.reduce_sum(tf.square(y_pred-Y))
    grads=tape.gradient(loos,variable)
    optimizer.apply_gradients(grads_and_vars=zip(grads,variable))
print(a,b)