import logging
import torch
import random
import pprint
#2
import model.XTDS.base
import util.tool

import torch.nn as nn
import numpy as np

from pytorch_transformers import BertTokenizer, BertModel, BertForMaskedLM, AdamW

from torch.nn import functional as F

class BERTTool(object):
    def init(args):
        BERTTool.multi_bert = BertModel.from_pretrained(args.multi_bert.location)
        BERTTool.multi_tokener = BertTokenizer.from_pretrained(args.multi_bert.location)
        BERTTool.multi_pad = BERTTool.multi_tokener.convert_tokens_to_ids(["[PAD]"])[0]
        BERTTool.multi_sep = BERTTool.multi_tokener.convert_tokens_to_ids(["[SEP]"])[0]
        BERTTool.multi_cls = BERTTool.multi_tokener.convert_tokens_to_ids(["[CLS]"])[0]
        #BERTTool.multi_bert.eval()
        #BERTTool.en_bert.eval()


class Model(model.XTDS.base.Model):
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
        self.iw = nn.Linear(768 ,len(self.ontology["intent"]))
        self.sw = nn.Linear(768 ,len(self.ontology["slot"]))

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
            ds = {"train": train, "dev": dev}
            ds.update(test)
            if not self.args.train.not_eval:
                for set_name, dataset in ds.items():
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

    def cross_list(self, x):
        return [self.cross(xx, not (self.training and self.args.train.ratio >= random.random())) for xx in x["utterance"]]

    def get_info(self, x):
        token_ids = []
        token_loc = []
        for xx in x:
            per_token_ids = [self.cls]
            per_token_loc = []
            per_type_ids = []
            per_mask_ids = []
            cur_idx = 1
            for token in xx:
                
                tmp_ids = self.tokener.encode(token)
                #print(token, tmp_ids)
                per_token_ids += tmp_ids
                per_token_loc.append(cur_idx)
                cur_idx += len(tmp_ids)
            per_token_ids += [self.sep]
            #print(per_token_ids)
            token_ids.append(per_token_ids)
            token_loc.append(per_token_loc)
        lens = [len(p) for p in token_ids]
        max_len = max(lens)
        mask_ids = []
        type_ids = []
        for per_token_ids in token_ids:
            per_mask_ids = [1] * len(per_token_ids) + [0] * (max_len - len(per_token_ids))
            per_token_ids += [self.pad] * (max_len - len(per_token_ids))
            per_type_ids = [0] * max_len
            mask_ids.append(per_mask_ids)
            type_ids.append(per_type_ids)
        token_ids = torch.Tensor(token_ids).long().to(self.device)
        mask_ids = torch.Tensor(mask_ids).long().to(self.device)
        type_ids = torch.Tensor(type_ids).long().to(self.device)
        return token_loc, token_ids, type_ids, mask_ids

    def forward(self, batch):
        token_loc, token_ids, type_ids, mask_ids = self.get_info([self.cross_list(x) for x in batch])
        h, utt = self.bert(token_ids, type_ids, mask_ids)
        outh = []
        for idx, locs in enumerate(token_loc):
             outh.append(h[idx][locs])
        outh = torch.cat(outh, dim=0)
        outh = self.sw(outh)
        outi = self.iw(utt)
        #print(outh.data.tolist())
        #print(torch.cat([torch.Tensor(x["slot"]) for x in batch], dim=0).long().data.tolist())
        loss = torch.Tensor([0])
        if self.training:
            loss = F.cross_entropy(outi, torch.Tensor(util.tool.in_each(batch, lambda x : x["intent"])).long().to(self.device)) + F.cross_entropy(outh, torch.cat([torch.Tensor(x["slot"]) for x in batch], dim=0).long().to(self.device))
        out = []
        loc_i = 0
        for idx, i in enumerate(outi):
            oo = outh[loc_i : loc_i + len(token_loc[idx])]
            loc_i += len(token_loc[idx])
            out.append((i.argmax().data.tolist(), [dd.argmax().data.tolist() for dd in oo]))
        return loss, out

    def start(self, inputs):
        train, dev, test, _, _, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        if self.args.model.resume is not None:
            self.run_eval(train, dev, test)
