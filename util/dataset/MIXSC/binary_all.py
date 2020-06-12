import os

import numpy as np
import random

import util.data
import util.convert
import util.tool

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix

class DatasetTool(object):

    def get_file(file, type):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        out = []
        for line in raw:
            out.append(tuple((line.lower(), type)))
        return out

    def get_set(file, args):
        mixed_file = os.path.join(file, "mixed.txt")
        neg_file = os.path.join(file, "neg.txt")
        pos_file = os.path.join(file, "pos.txt")
        sneg_file = os.path.join(file, "strneg.txt")
        spos_file = os.path.join(file, "strpos.txt")
        sneg = DatasetTool.get_file(sneg_file, 0);
        neg = DatasetTool.get_file(neg_file, 0);
        mixed = []
        pos = DatasetTool.get_file(pos_file, 1);
        spos = DatasetTool.get_file(spos_file, 1);
        dataset = sneg + neg + mixed + pos + spos
        random.shuffle(dataset)
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
        """
        Get train, dev, test set from files by args.

        Inputs:
            args: Arguments

        Returns:
            train, dev, test: Train, dev, test data sets.
            ontology: ontology file of WOZ
            emb: Embedding of data
            vocab: Dict of the embedding

        Examples:
            get(args) => (train, dev, test, ontology, emb, vocab)
        """
        train_file = os.path.join(args.dir.dataset, args.dataset.src, "train")
        dev_file = os.path.join(args.dir.dataset, args.dataset.src, "dev")
        args.dict_list = args.dataset.dict.split(" ")
        args.test_list = args.dataset.tgt.split(" ")
        train = DatasetTool.get_set(train_file, args)
        dev = DatasetTool.get_set(dev_file, args)
        test = {test: DatasetTool.get_set(os.path.join(args.dir.dataset, test, "test"), args) for test in args.test_list}
        idx_dict = util.convert.Common.to_args({"src2tgt": []})
        for dict_file in args.dict_list:
            dict_file = os.path.join(args.dir.dataset, dict_file)
            DatasetTool.get_idx_dict(idx_dict, dict_file, args)
        return train, dev, test, None, idx_dict, None

    def evaluate2(pred, dataset, args):
        num = 0
        cor = 0
        precision = np.zeros(args.train.level)
        recall = np.zeros(args.train.level)
        f1 = np.zeros(args.train.level)
        confusion = np.zeros((args.train.level, args.train.level))
        for p, l in zip(pred, dataset):
            confusion[p][l[1]] += 1
            num += 1
            if p == l[1]:
                cor += 1
        for i in range(args.train.level):
            precision[i] = confusion[i][i] / np.sum(confusion, axis = 0)[i]
            recall[i] = confusion[i][i] / np.sum(confusion, axis = 1)[i]
        return {'accuracy': cor / num, "recall": np.mean(recall), "precision": np.mean(precision), "f1": 2*np.mean(recall)*np.mean(precision)/(np.mean(recall)+np.mean(precision))}
        
    def record(pred, dataset, set_name, args):
        pass

    def evaluate(pred, dataset, args):
        label = []
        for line in dataset:
            label.append(line[1])
        label = label[:len(pred)]
        acc = accuracy_score(label, pred)
        prec = precision_score(label, pred,  average='binary')
        rec = recall_score(label, pred, average='binary')
        f1 = f1_score(label, pred,  average='binary')
        return {'accuracy': acc, "recall": rec, "precision": prec, "f1": f1}