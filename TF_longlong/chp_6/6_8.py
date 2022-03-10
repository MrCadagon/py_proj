# coding=utf-8
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets
from tensorflow.keras import Sequential
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def load_data():
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    # MAP: 每加仑英里数
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    # read_csv参数名称:
    # na_values: scalar, str, list - like, or dict, default,None
    # 一组用于替换NA / NaN的值。
    # comment: str, default,None
    # 标识着多余的行不被解析。如果该字符出现在行首，这一行将被全部忽略。
    # sep: str, default ‘, ’
    # 指定分隔符。如果不指定参数，则会尝试使用逗号分隔。csv文件一般为逗号分隔符。

    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)
    return raw_dataset


def preprocess_dataset(dataset):
    dataset = dataset.copy()
    dataset = dataset.dropna()
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0

    # frac:比例
    train_dataset = dataset.sample(frac=0.8)
    test_dataset = dataset.drop(train_dataset.index)
    return dataset, test_dataset


#创建netwok
class Network(keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = layers.Dense(64, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, input):
        x=self.fc1(input)
        x=self.fc2(x)
        x=self.fc3(x)
        return x


# def build_model():
#     model = Network()    # 创建网络
#     model.build(input_shape=(4, 9))# 通过build 函数完成内部张量的创建，其中4 为任意设置的batch 数量，9 为输入特征长度
#     model.summary()# 打印网络信息
#     return model
# model = build_model() #创建模型
# optimizer = tf.keras.optimizers.RMSprop(0.001)# 创建优化器，指定学习率
# train_db = tf.data.Dataset.from_tensor_slices((normed_train_data.values, train_labels.values))
# train_db = train_db.shuffle(100).batch(32)


dataset_retured = load_data()
train_dataset, test_dataset = preprocess_dataset(dataset_retured)
sns_plot = sns.pairplot(train_dataset[["Cylinders", "Displacement", "Weight", "MPG"]], diag_kind="kde")
plt.show()

print(dataset_retured)
print(train_dataset)
print(test_dataset)

print('end')
