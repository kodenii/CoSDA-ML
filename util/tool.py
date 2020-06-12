import importlib
import random

class Args(object):
    def __init__(self, contain = None):
        self.__self__ = contain
        self.__default__ = None
        self.__default__ = set(dir(self))

    def __call__(self):
        return self.__self__

    def __getattribute__(self, name):
        if name[:2] == "__" and name [-2:] == "__":
            return super().__getattribute__(name)
        if name not in dir(self):
            return None
        return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if value != None or (name[:2] == "__" and name [-2:] == "__"):
            return super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in dir(self) and name not in self.__default__:
            super().__delattr__(name)

    def __iter__(self):
        return list((arg, getattr(self, arg)) for arg in set(dir(self)) - self.__default__).__iter__()

    def __len__(self):
        return len(set(dir(self)) - self.__default__)

class Vocab(object):
    def __init__(self, words, add_pad = False):
        if add_pad:
            self.wordlist = ["<unk>"] + ["<sos>"] + ["<eos>"] + words
        else:
            self.wordlist = words
        self.worddict = {}
        for idx, word in enumerate(self.wordlist):
            self.worddict[word] = idx

    def __len__(self):
        return len(self.wordlist)

    def __iter__(self):
        return self.wordlist.__iter__()

    def word2idx(self, word):
        if word not in self.wordlist:
            return self.worddict["<unk>"]
        return self.worddict[word]

    def idx2word(self, idx):
        return self.wordlist[idx]

class Batch(object):
    def to_list(source, batch_size):
        batch_list = []
        idx = 0
        while idx < len(source):
            next_idx = idx + batch_size
            if next_idx > len(source):
                next_idx = len(source)
            batch_list.append(source[idx : next_idx])
            idx = next_idx
        return batch_list

    def get_batch(source, batch_size, idx):
        bgn = min(idx * batch_size, len(source))
        end = min((idx + 1) * batch_size, len(source))
        return source[bgn : end]

def idx_extender(source, max_len = None, pad = None, bias = 0):
    out = []
    for idx, num in source:
        for _ in range(num):
            out.append(idx + bias)
    cur_len = len(out)
    while cur_len < max_len:
        out.append(pad)
        cur_len += 1
    return out

def in_each(source, method, cond = True):
    return [method(x) for x in source if cond or cond(x)]

def pad(inputs, pad):
    max_len = 0
    for input in zip(*inputs):
        cur_len = 0
        for x in input:
            cur_len += len(x)
        if cur_len > max_len:
            max_len = cur_len
    all_padded = []
    all_idx = []
    for input in zip(*inputs):
        line_padded = []
        line_idx = []
        for idx, x in enumerate(input):
            line_idx.append((idx, len(x)))
            line_padded += x
            idx += 1
        cur_len = len(line_padded)
        while cur_len < max_len:
            line_padded.append(pad)
            cur_len += 1
        all_padded.append(line_padded)
        all_idx.append(line_idx)
    return all_padded, all_idx
        

def load_module(model, dataset_tool):
    Model = importlib.import_module('model.{}'.format(model)).Model
    DatasetTool = importlib.import_module('util.dataset.{}'.format(dataset_tool)).DatasetTool
    return Model, DatasetTool