from collections import Counter
import numpy as np

class Vocab(object):

    PAD = 0
    SOS = 1
    EOS = 2
    UNK = 3

    def __init__(self):
        self.word2index = {} # 根据 word 得到对应的 index
        self.word2count = Counter() # 根据 word 得到该词的频率
        self.reserved = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] # 默认字符
        self.index2word = self.reserved[:] # 根据 index 得到对应的 word
        self.embeddings = None

    def add_words(self, words):
        """
        :param words: 单词序列
        :return: 更新 word2index，index2word，word2count
        """
        for word in words:
            if word not in self.word2index:
                self.word2index[word] = len(self.index2word)
                self.index2word.append(word)
        self.word2count.update(words)

    def load_embeddings(self, file_path: str, dtype=np.float32) -> int:
        num_embeddings = 0
        vocab_size = len(self)
        with open(file_path, 'rb') as f:
            for line in f:
                line = line.split()
                word = line[0].decode('utf-8')
                idx = self.word2index.get(word)
                if idx is not None:
                    vec = np.array(line[1:], dtype=dtype)
                    if self.embeddings is None:
                        n_dims = len(vec)
                        self.embeddings = np.random.normal(
                            np.zeros((vocab_size, n_dims))).astype(dtype)
                        self.embeddings[self.PAD] = np.zeros(n_dims)
                    self.embeddings[idx] = vec
                    num_embeddings += 1
        return num_embeddings

    def __getitem__(self, item):
        """
        定义这个函数可以通过：实例化 Vocab 然后 vocab[5] 的方式取值
        """
        if type(item) is int:
            return self.index2word[item]
        return self.word2index.get(item, self.UNK)

    def __len__(self):
        return len(self.index2word)

    def size(self):
        """Returns the total size of the vocabulary"""
        return len(self.index2word)
