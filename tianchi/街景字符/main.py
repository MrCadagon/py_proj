# https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.2ce879deGunCpk&postId=108659#
# https://tianchi.aliyun.com/competition/entrance/531795/information

import json
import numpy as np
import matplotlib.pyplot as plt
import cv2

train_json = json.load(open('./file_josn/111.json'))


# Task1
# 数据标注处理
def parse_json(d):
    arr = np.array([
        d['top'], d['height'], d['left'], d['width'], d['label']
    ])
    arr = arr.astype(int)
    return arr


img = cv2.imread('./mchar_train/000000.png')
arr = parse_json(train_json['000000.png'])
plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1] + 1, 1)
plt.imshow(img)
plt.xticks([]);
plt.yticks([])
for idx in range(arr.shape[1]):
    plt.subplot(1, arr.shape[1] + 1, idx + 2)
    plt.imshow(img[arr[0, idx]:arr[0, idx] + arr[1, idx], arr[2, idx]:arr[2, idx] + arr[3, idx]])
    plt.title(arr[4, idx])
    plt.xticks([]);
    plt.yticks([])
plt.show()

# Task2
# 图像读取
from PIL import Image, ImageFilter

plt.figure(figsize=(10, 10))
plt.subplot(1, 1, 1)
im = Image.open('./mchar_train/000000.png')
plt.imshow(im)
plt.show()

# 应用模糊滤镜
im_filter = im.filter(ImageFilter.BLUR)
plt.imshow(im_filter)
plt.show()

# 高度和宽度
plt.figure(figsize=(10, 10))
# im3 = im.thumbnail((im.width / 2, im.height / 2))
print('aa')
# plt.imshow(im3)
# plt.show()

# OpenCV  灰度图 边缘检测
img = cv2.imread('./cat.jpg')
plt.imshow(img)
plt.title("GRAY")
plt.show()
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title("GRAY")
plt.show()
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img3, cmap='gray')
plt.title("GRAY3")
plt.show()
# Canny边缘检测
edges = cv2.Canny(img, 30, 70)
plt.imshow(edges, cmap='gray')
plt.show()


# Task3 字符识别模型

