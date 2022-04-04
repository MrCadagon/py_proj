# 参考https://www.manongdao.com/article-188629.html
import numpy as np
import torch


# 全连接层
class fc():
    def __init__(self):
        self.w = np.array([0.01])
        self.b = np.array([0.01])

    def forward(self, input):
        self.input = input
        output = np.dot(self.input, self.w) + self.b
        return output

    def gra(self, err):
        for i in range(err.shape[0]):
            col_input = self.input[i][:, np.newaxis]
            err_i = err[i][:, np.newaxis].T
            self.w_gra += np.dot(col_input, err_i)
            self.b_gra += err_i.reshape(self.b.shape)
        next_err = np.dot(err, self.w.T)
        next_err = np.reshape(next_err, self.input_shape)
        return next_err

    def backward(self, err, alpha=0.00001, weight_decay=0.0001):
        next_err = self.gra(err)
        self.w *= (1 - weight_decay)
        self.b *= (1 - weight_decay)
        self.w -= alpha * self.w_gra
        self.b -= alpha * self.b_gra

        self.w_gra = np.zeros(self.w.shape)
        self.b_gra = np.zeros(self.b.shape)

        return next_err


model = fc()
criterion = torch.nn.MSELoss(size_average=False)

# x_data = torch.Tensor([[1.0], [2.0], [3.0]])
x_data = np.array([[1.0], [2.0], [3.0]])
# y_data = torch.Tensor([[2.0], [4.0], [6.0]])
y_data = np.array([[2.0], [4.0], [6.0]])

for epoch in range(100):
    y_pred = model.forward(x_data)
    print(y_pred.resize(1, -1))
    print(y_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss)

    # optimizer.zero_grad()
    model.backward(loss)
    # optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred', y_test.data)

# conv
# class conv_a():
#     def forward(self, input):
#
#         self.input = input
#         col_w = self.w.reshape([-1, self.output_channels])  # 每一层权重转换为向量
#         self.col_image = []
#         conv_out = np.zeros(self.output_shape)
#
#         for i in range(self.batchsize):
#             img_i = self.input[i][np.newaxis, :]  # 新的轴
#             self.col_img_i = img2col(img_i, self.kernal_size, self.stride)
#             conv_out[i] = np.reshape(np.dot(self.col_img_i, col_w) + self.b, self.err[0].shape)  # eerr[0]是因为这里是单个图片
#             self.col_image.append(self.col_img_i)
#         self.col_image = np.array(self.col_image)
#
#         return conv_out
#
#     def gra(self, err):
#
#         self.err = err
#         col_err = np.reshape(err, [self.batchsize, -1, self.output_channels])
#         for i in range(self.batchsize):
#             self.w_gra += np.dot(self.col_image[i].T, col_err[i]).reshape(self.w.shape)
#         self.b_gra += np.sum(col_err, axis=(0, 1))
#
#         pad_err = np.pad(self.err,
#                          ((0, 0), (self.kernal_size - 1, self.kernal_size - 1),
#                           (self.kernal_size - 1, self.kernal_size - 1), (0, 0)),
#                          'constant', constant_values=0)
#
#         flip_w = np.flipud(np.fliplr(self.w))
#         flip_w = flip_w.swapaxes(2, 3)
#         col_flip_w = flip_w.reshape([-1, self.input_channels])
#         col_pad_err = np.array(
#             [img2col(pad_err[i][np.newaxis, :], self.kernal_size, self.stride) for i in range(self.batchsize)])
#         next_err = np.dot(col_pad_err, col_flip_w)
#         next_err = np.reshape(next_err, self.input_shape)
#
#         return next_err
#
#     def backward(self, err, alpha=0.00001, weight_decay=0.0001):
#
#         next_err = self.gra(err)
#         self.w *= (1 - weight_decay)
#         self.bias = (1 - weight_decay)
#         self.w -= alpha * self.w_gra
#         self.b -= alpha * self.b_gra
#
#         self.w_gra = np.zeros(self.w.shape)
#         self.b_gra = np.zeros(self.b.shape)
#
#         return next_err
