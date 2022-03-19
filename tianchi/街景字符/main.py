# https://tianchi.aliyun.com/notebook-ai/detail?spm=5176.12586969.1002.15.2ce879deGunCpk&postId=108659#
# https://tianchi.aliyun.com/competition/entrance/531795/information
import scipy.io as scio
import mat73
dataFile = '/home/k/下载/train/digitStruct.mat'

print('before')
data_dict = mat73.loadmat(dataFile)
print('after')