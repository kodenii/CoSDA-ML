import logging
import torch
import random
import pprint

import model.MLDoc.base
import util.tool

import torch.nn as nn
import numpy as np

from transformers import XLMTokenizer, XLMModel, AdamW

from torch.nn import functional as F

class BERTTool(object):
    def init(args):
        BERTTool.multi_bert = XLMModel.from_pretrained(args.multi_bert.location)
        BERTTool.multi_tokener = XLMTokenizer.from_pretrained(args.multi_bert.location)
        BERTTool.multi_pad = BERTTool.multi_tokener.convert_tokens_to_ids(["<pad>"])[0]
        BERTTool.multi_sep = BERTTool.multi_tokener.convert_tokens_to_ids(["</s>"])[0]
        BERTTool.multi_cls = BERTTool.multi_tokener.convert_tokens_to_ids(["<s>"])[0]
        #BERTTool.multi_bert.eval()
        #BERTTool.en_bert.eval()


class Model(model.MLDoc.base.Model):
    def __init__(self, args, DatasetTool, inputs):
        np.random.seed(args.train.seed)
        torch.manual_seed(args.train.seed)
        random.seed(args.train.seed)
        super().__init__(args, DatasetTool, inputs)
        _, _, _, ontology, worddict, _ = inputs
        self.ontology = ontology
        self.worddict = worddict
        BERTTool.init(self.args)
        self.bert = BERTTool.multi_bert
        self.tokener = BERTTool.multi_tokener
        self.pad = BERTTool.multi_pad
        self.sep = BERTTool.multi_sep
        self.cls = BERTTool.multi_cls
        self.P = torch.nn.Linear(1024, args.train.level)
        # W will not be used in this version
        self.W = torch.nn.Linear(1024, 1024)
        self.Loss = torch.nn.CrossEntropyLoss()

    def set_optimizer(self):
        all_params = set(self.parameters())
        if self.args.train.bert == False:
            bert_params = set(BERTTool.multi_bert.parameters())
            for para in bert_params:
                para.requires_grad=False
            params = [{"params": list(all_params - bert_params), "lr": self.args.lr.default}]
        else:
            bert_params = set(BERTTool.multi_bert.parameters())
            params = [{"params": list(all_params - bert_params), "lr": self.args.lr.default},
                      {"params": list(bert_params), "lr": self.args.lr.bert}
                      ]
        self.optimizer = AdamW(params)

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        self.eval()
        summary = {}
        ds = {"train": train, "dev": dev}
        ds.update(test)
        for set_name, dataset in ds.items():
            self.args.need_w = False
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.ontology = self.ontology
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            iteration += iter
            summary.update({"loss": loss})
            if not self.args.train.not_eval:
                ds = {"train": train, "dev": dev}
                ds.update(test)
                for set_name, dataset in ds.items():
                    self.args.need_w = False
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

    def cross(self, x, disable=False):
        if not disable and self.training and (self.args.train.cross >= random.random()):
            lan = random.randint(0,len(self.args.dict_list) - 1)
            if x in self.worddict.src2tgt[lan]:
                return self.worddict.src2tgt[lan][x][random.randint(0,len(self.worddict.src2tgt[lan][x]) - 1)]
            else:
                return x
        else:
            return x

    def cross_str(self, x, disable=False):
        raw = x.lower().split(" ")
        out = ""
        for xx in raw:
            out += self.cross(xx, disable)
            out += " "
        #print(out)
        return out

    def cross_list(self, x, disable=False):
        return [self.cross_str(xx, not (self.training and self.args.train.ratio >= random.random())) for xx in x]

    def forward(self, batch):
        a,b,c = util.convert.List.to_bert_info(self.cross_list(util.tool.in_each(batch, lambda x : x[0])), self.tokener, self.pad, self.cls, self.device, 128)[0]
        h = self.bert(a,c)[0]
        utt = h[:,0]
        out = self.P(utt)
        loss = torch.Tensor([0])
        if self.training:
            label = util.tool.in_each(batch, lambda x : x[1])
            loss = self.Loss(out, torch.Tensor(label).long().to(self.device))
        return loss, out

    def get_pred(self, out):
        return torch.argmax(out, dim = 1).tolist()

    def start(self, inputs):
        train, dev, test, _, _, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if self.args.model.w is not None:
            self.load_w(self.args.model.w)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        if self.args.model.resume is not None:
            self.run_eval(train, dev, test)

    def load_w(self, file):
        logging.info("Loading w from {}".format(file))
        state = torch.load(file)
        new_state = {"bias": torch.zeros(self.args.dimension.emb), "weight": state["weight"]}
        self.W.load_state_dict(new_state)

    def load(self, file):
        logging.info("Loading model from {}".format(file))
        state = torch.load(file)
        model_state = state["model"]
        model_state.update({"W.weight": self.W.weight, "W.bias": self.W.bias})
        self.load_state_dict(model_state)