# 参考：https://blog.csdn.net/weixin_37251044/article/details/81349287

#5.4.2 卷积层逐行推导
import numpy as np
relu1_data = np.load('./data_pic/data_3_relu1[10-32-14-14].npy')

print ("卷积层的输入的shape是：\n"+str(relu1_data.shape))

pad = 1

in_data_pad = np.pad(relu1_data, ((0,), (0,), (pad,), (pad,)), mode='constant', constant_values=0) #边缘填充，这里上下左右各填一个像素

print ("\n卷积层的输入边缘填充之后的shape是：\n"+str(in_data_pad.shape))

print ("\n卷积层的输入边缘填充之后的第一个batch的第一个channel的前三行是：\n"+str(in_data_pad[0][0][0:3][0:3]))
