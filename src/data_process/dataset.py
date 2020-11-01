import sys
import os
import pathlib
from collections import Counter
from typing import Callable

import torch
from torch.utils.data import Dataset

abs_path = pathlib.Path(__file__).parent.absolute()
sys.path.append(sys.path.append(abs_path))

from src.utils.util import simple_tokenizer, count_words, sort_batch_by_len, source2ids, abstract2ids
from src.data_process.vocab import Vocab
import src.utils.config as config


class PairDataset(object):
    """
    原始数据集数据格式为：鸭 服饰 雪纺 连衣裙 女 夏季 2019 新款 ... <sep> ... 因此 即便 是 黑色 也 毫不 沉闷 仙气 十足 。
    source<sep>target
    返回： (['犀牛', '褶', '抗皱', '红豆', '西裤', '男', '修身', '冬季', '正装', '商务'... ], ['下身', '搭配', '西装裤', '和', '牛津', '鞋', '，', '时髦', '又', '不会', '过时', '。'])
    """
    def __init__(self,
                 filename, # train.txt
                 tokenize: Callable = simple_tokenizer,
                 max_src_len: int = None, # 300
                 max_tgt_len: int = None, # 100
                 truncate_src: bool = False, # 当 source text 大于 max_src_len，是否截断
                 truncate_tgt: bool = False): # 当 reference text 大于 max_tgt_len，是否截断

        print("Reading dataset %s..." % filename, end=' ', flush=True)

        self.filename = filename
        self.pairs = []

        with open(filename, 'rt', encoding='utf-8') as f: # rt模式下，Python 在读取文本时会自动把\r\n转换成\n，文本文件用二进制读取用‘rt’；
            # next(f)
            for i, line in enumerate(f):
                # line: 鸭 服饰 雪纺 连衣裙 女 夏季 2019新款 裙子 女装 夏装 韩版件 面料 其它 ... <sep>层次感 薄纱 连衣裙 想要 不 膨胀 首先 面料 要 薄 ， 这 条 黑色 蛋糕 款 材质 轻若 无物 ， 其次 露出 单肩 、 收出 腰线 都 能 为 整体 减重 ， 此外 网纱雪纺 层次 疏朗 也 有效 控制 了 膨胀 度 ， 因此 即便 是 黑色 也 毫不 沉闷 仙气 十足 。
                # Split the source and reference by the <sep> tag.
                pair = line.strip().split('<sep>')
                if len(pair) != 2:
                    print("Line %d of %s is malformed." % (i, filename))
                    print(line)
                    continue
                # pair[0]: source
                # pair[1]: reference
                src = tokenize(pair[0])
                if max_src_len and len(src) > max_src_len:
                    if truncate_src:
                        src = src[:max_src_len]
                    else:
                        continue
                tgt = tokenize(pair[1])
                if max_tgt_len and len(tgt) > max_tgt_len:
                    if truncate_tgt:
                        tgt = tgt[:max_tgt_len]
                    else:
                        continue
                self.pairs.append((src, tgt))

        # pairs[0]: (['犀牛', '褶', '抗皱', '红豆', '西裤', '男', '修身', '冬季', '正装', '商务'... ], ['看似', '单调', '的', '纯色', '毛衣', '，', '内', '搭', '一件', '格子', '衬衫', '立刻', '让', '整体', '穿', '搭', '散发出', '帅气', '的', '味道', '，', '下身', '搭配', '西装裤', '和', '牛津', '鞋', '，', '时髦', '又', '不会', '过时', '。'])
        print("%d pairs." % len(self.pairs))

    def build_vocab(self, embed_file: str = None) -> Vocab:
        word_counts = Counter()
        count_words(word_counts, [src + tgr for src, tgr in self.pairs])
        # 构建了 word：frequncy 的 Counter
        vocab = Vocab()
        # Filter the vocabulary by keeping only the top k tokens in terms of
        # word frequncy in the data set, where k is the maximum vocab size set
        # in "config.py".
        for word, count in word_counts.most_common(config.max_vocab_size):
            vocab.add_words([word])
        if embed_file is not None:
            count = vocab.load_embeddings(embed_file)
            print("%d pre-trained embeddings loaded." % count)

        return vocab


class SampleDataset(Dataset):
    """The class represents a sample set for training."""
    def __init__(self, data_pair, vocab):
        self.src_sents = [x[0] for x in data_pair]
        self.trg_sents = [x[1] for x in data_pair]
        self.vocab = vocab
        # Keep track of how many data points.
        self._len = len(data_pair)

    def __getitem__(self, index):
        x, oov = source2ids(self.src_sents[index], self.vocab)
        return {
            'x': [self.vocab.SOS] + x + [self.vocab.EOS],
            'OOV': oov,
            'len_OOV': len(oov),
            'y': [self.vocab.SOS] + abstract2ids(self.trg_sents[index], self.vocab, oov) + [self.vocab.EOS],
            'x_len': len(self.src_sents[index]),
            'y_len': len(self.trg_sents[index])
        }

    def __len__(self):
        return self._len


def collate_fn(batch):
    """Split data set into batches and do padding for each batch.
    Args:
        x_padded (Tensor): Padded source sequences.
        y_padded (Tensor): Padded reference sequences.
        x_len (int): Sequence length of the sources.
        y_len (int): Sequence length of the references.
        OOV (dict): Out-of-vocabulary tokens.
        len_OOV (int): Number of OOV tokens.
    """
    print(batch)
    def padding(indice, max_length, pad_idx=0):
        pad_indice = [item + [pad_idx] * max(0, max_length - len(item)) for item in indice]
        return torch.tensor(pad_indice)

    data_batch = sort_batch_by_len(batch)

    x = data_batch["x"]
    x_max_length = max([len(t) for t in x])
    y = data_batch["y"]
    y_max_length = max([len(t) for t in y])

    OOV = data_batch["OOV"]
    len_OOV = torch.tensor(data_batch["len_OOV"])

    x_padded = padding(x, x_max_length)
    y_padded = padding(y, y_max_length)

    x_len = torch.tensor(data_batch["x_len"])
    y_len = torch.tensor(data_batch["y_len"])
    return x_padded, y_padded, x_len, y_len, OOV, len_OOV

