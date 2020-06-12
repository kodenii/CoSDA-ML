import os

import numpy as np

import util.data
import util.convert
import util.tool

class DatasetTool(object):

    def get_set(file):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        dataset = []
        for line in raw:
            line = line.split("\t")
            dataset.append(tuple((line[1], int(line[0]))))
        return dataset

    def get_idx_dict(idx_dict, file, args):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        if args.dataset.part is not None:
            raw = raw[ : args.dataset.part * 100]
        idx_dict.src2tgt.append({})
        for line in raw:
            try:
                src, tgt = line.split("\t")
            except:
                src, tgt = line.split(" ")
            
            if src not in idx_dict.src2tgt[-1]:
                idx_dict.src2tgt[-1][src] = [tgt]
            else:
                idx_dict.src2tgt[-1][src].append(tgt)

    def get(args):
        train_file = os.path.join(args.dir.dataset, args.dataset.train)
        dev_file = os.path.join(args.dir.dataset, args.dataset.dev)
        args.dict_list = args.dataset.dict.split(" ")
        args.test_list = args.dataset.test.split(" ")
        train = DatasetTool.get_set(train_file)
        dev = DatasetTool.get_set(dev_file)
        test = {test: DatasetTool.get_set(os.path.join(args.dir.dataset, test)) for test in args.test_list}
        idx_dict = util.convert.Common.to_args({"src2tgt": []})
        for dict_file in args.dict_list:
            dict_file = os.path.join(args.dir.dataset, dict_file)
            DatasetTool.get_idx_dict(idx_dict, dict_file, args)
        return train, dev, test, None, idx_dict, None

    def evaluate(pred, dataset, args):
        summary = {}
        correct = 0
        all = 0
        for p, g in zip(pred, dataset):
            all += 1
            if p == g[1]:
                correct += 1
        summary.update({"accuracy": correct / all})
        return summary

    def record(pred, dataset, set_name, args):
        pass
