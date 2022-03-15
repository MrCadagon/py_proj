import torch
import random
import zipfile

with zipfile.ZipFile('./jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])
# 简化
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')
corpus_chars = corpus_chars[0:10000]

idx_to_char = list(set(corpus_chars))
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])
print(char_to_idx)
vocab_size = len(char_to_idx)
print(vocab_size)  # 1027
# 将训练数据集中每个字符转化为索引
corpus_indices = [char_to_idx[c] for c in corpus_chars]
sample = corpus_indices[:20]
print('\n', 'chars:', ''.join([idx_to_char[i] for i in sample]), '\n')
print('indices',sample)


# 随机采样:随机采样一个小批量   每个样本所包含的时间步数
# 相邻的两个随机小批量在原始序列上的位置不一定相毗邻
# 无法用一个小批量最终时间步的隐藏状态来初始化下一个小批量的隐藏状态。在训练模型时，每次随机采样前都需要重新初始化隐藏状态。
