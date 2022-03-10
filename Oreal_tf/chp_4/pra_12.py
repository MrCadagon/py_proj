# Python ≥3.5 is required
import sys

assert sys.version_info >= (3, 5)
# Scikit-Learn ≥0.20 is required
import sklearn

assert sklearn.__version__ >= "0.20"
# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
# To plot pretty figures
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


from sklearn import datasets

# 1.获取数据
iris = datasets.load_iris()
list(iris.keys())
X = iris["data"][:, (2, 3)]  # petal length, petal width
y = iris["target"]
# print(X)


# 2.add bias 获取train test valid数据集
np.random.seed(2042)
X_with_bias = np.c_[np.ones([len(X), 1]), X]

source_size = len(X)
test_ratio = 0.2
test_size = int(source_size * test_ratio)
valid_ratio = 0.2
valid_size = int(source_size * valid_ratio)
train_size = source_size - test_size - valid_size
rnd_indices_shuffle = np.random.permutation(source_size)

X_train = X_with_bias[rnd_indices_shuffle[:train_size]]
Y_train = y[rnd_indices_shuffle[:train_size]]
X_test = X_with_bias[rnd_indices_shuffle[train_size:-test_size]]
Y_test = y[rnd_indices_shuffle[train_size:-test_size]]
X_valid = X_with_bias[rnd_indices_shuffle[-test_size:]]
Y_valid = y[rnd_indices_shuffle[-test_size:]]


# 3预处理数据
def softmax(logits):
    exps = np.exp(logits)
    exp_sum = np.sum(exps, axis=1, keepdims=True)
    print(exp_sum.shape)
    print(exps.shape)
    return exps / exp_sum

def to_one_hot(y):
    m = len(y)
    n = y.max() + 1
    y_one_shot = np.zeros((m, n))
    y_one_shot[np.arange(m), y] = 1
    return y_one_shot


# 4 开始优化
Y_train_one_shot = to_one_hot(Y_train)
Y_valid_one_shot = to_one_hot(Y_valid)
Y_test_one_shot = to_one_hot(Y_test)

print('end')
