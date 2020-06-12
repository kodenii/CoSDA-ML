import logging
import torch
import random
import pprint
import os
from tqdm import tqdm
#2
import model.DST.base
import util.tool

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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


class Model(model.DST.base.Model):
    def __init__(self, args, DatasetTool, inputs):
        super().__init__(args, DatasetTool, inputs)
        _, _, _, (ontology_src, ontology_tgt_it, ontology_tgt_de), worddict, para = inputs
        self.ontology_src = ontology_src
        self.ontology_tgt = [ontology_tgt_it, ontology_tgt_de]
        self.ontology = self.ontology_src
        self.worddict = worddict
        self.w = nn.Linear(768 * 2, 1)
        self.para = para
        BERTTool.init(self.args)
        self.bert = BERTTool.multi_bert
        self.tokener = BERTTool.multi_tokener
        self.pad = BERTTool.multi_pad
        self.sep = BERTTool.multi_sep
        self.cls = BERTTool.multi_cls
        for s in self.ontology.slots:
            setattr(self, '{}_l'.format(s), nn.Linear(768*2 ,len(self.ontology.values[s])))

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

    def visualize(self, epoch=-1):
        return
        self.bert.eval()
        all = None
        for line in tqdm(self.para):
            _, utt_src = self.bert(*util.convert.List.to_bert_info2(util.tool.in_each([line], lambda x : x[0]), [" "]*1, self.tokener, self.pad, self.cls, self.sep, self.device)[0])
            if all is None:
                all = utt_src.detach().cpu().numpy()
            else:
                all = np.concatenate((all, utt_src.detach().cpu().numpy()))
        for line in tqdm(self.para):
            _, utt_src = self.bert(*util.convert.List.to_bert_info2(util.tool.in_each([line], lambda x : x[1]), [" "]*1, self.tokener, self.pad, self.cls, self.sep, self.device)[0])
            if all is None:
                all = utt_src.detach().cpu().numpy()
            else:
                all = np.concatenate((all, utt_src.detach().cpu().numpy()))
        sne = TSNE(n_components=2, init='pca').fit_transform(all)
        fig = plt.figure(epoch)
        for x,y,c,m in zip(sne[:, 0], sne[:, 1], ['r']*len(self.para)+['g']*len(self.para), ["o"]*len(self.para)*2):
            plt.scatter([x], [y], c=[c], marker=m)
        if not os.path.exists(self.args.dir.output):
            os.makedirs(self.args.dir.output)
        plt.savefig(os.path.join(self.args.dir.output, str(epoch)+".pdf"))
        plt.close(epoch)


    def run_train(self, train, dev, test):
        self.set_optimizer()
        test_it, test_de = test
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.ontology = self.ontology_src
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            iteration += iter
            summary.update({"loss": loss})
            if not self.args.train.not_eval:
                for set_name, dataset in {"train": train, "dev": dev, "test_it": test_it, "test_de": test_de}.items():
                    if set_name == "test_it":
                        self.ontology = self.ontology_tgt[0]
                    elif set_name == "test_de":
                        self.ontology = self.ontology_tgt[1]
                    else:
                        self.ontology = self.ontology_src
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
            best = self.update_best(best, summary, epoch)
            #self.visualize(epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))
        self.visualize(self.args.train.epoch)
    
    def to_glad(self, raw):
        self.bert.eval()
        input, lens = util.convert.List.to_bert_info(raw, self.tokener, self.pad, self.cls, self.device)
        out, _ = self.bert(*input, output_all_encoded_layers=False)
        return out, lens

    def cross(self, x, disable=False):
        lan = random.randint(0,1)
        if not disable and self.training and (x in self.worddict.src2tgt[lan] and self.args.train.cross >= random.random()):
            return self.worddict.src2tgt[lan][x][random.randint(0,len(self.worddict.src2tgt[lan][x]) - 1)]
        else:
            return x

    def cross_str(self, x, disable=False):
        raw = x.lower().split(" ")
        out = ""
        for xx in raw:
            out += self.cross(xx, disable)
            out += " "
        return out

    def cross_list(self, x):
        return [self.cross_str(xx, not (self.training and self.args.train.ratio >= random.random())) for xx in x]

    def cross_value(self, s, x):
        if isinstance(x, list):
            return [self.cross_value(s, xx) for xx in x]
        elif self.training and self.args.train.cross >= random.random():
            lan = random.randint(0,1)
            return self.ontology_tgt[lan].values[s][self.ontology.values[s].index(x)]
        else:
            return x

    def cross_slot(self, x):
        if self.training and self.args.train.cross >= random.random():
            lan = random.randint(0,1)
            return self.ontology_tgt[lan].slots[self.ontology.slots.index(x)]
        return x

    def revert_slot(self, x):
        if not self.training:
            try:
                return self.ontology_src.slots[self.ontology_tgt[0].slots.index(x)]
            except:
                return self.ontology_src.slots[self.ontology_tgt[1].slots.index(x)]
        return x

    def forward(self, batch):
        # convert to variables and look up embeddings
        if self.args.train.bert == False:
            self.bert.eval()
        tmp = util.convert.List.to_bert_info2(self.cross_list(util.tool.in_each(batch, lambda x : x.transcript)), self.cross_list(util.tool.in_each(batch, lambda x : util.convert.List.to_str(x.system_acts))), self.tokener, self.pad, self.cls, self.sep, self.device)
        _, utt = self.bert(*tmp[0])

        ys = {}
        for s in self.ontology.slots:
            if s != self.args.train.slot and self.args.train.slot is not None:
                continue
            # for each slot, compute the scores for each value
            if self.args.train.bert == False:
                self.bert.eval()
            _, sv = self.bert(*util.convert.List.to_bert_info2([self.cross_slot(s)], self.cross_value(s, self.ontology.values[s]), self.tokener, self.pad, self.cls, self.sep, self.device)[0])
            l = getattr(self, '{}_l'.format(self.revert_slot(s)))
            cls = torch.cat((utt, sv.expand(utt.shape[0],768)),dim=1)
            out = l(cls)
            # combine the scores
            ys[s] = F.sigmoid(out)

        if self.training:
            # create label variable and compute loss
            labels = {s: [len(self.ontology.values[s]) * [0] for i in range(len(batch))] for s in self.ontology.slots}
            for i, e in enumerate(batch):
                for s, v in e.turn_label:
                    if s != self.args.train.slot and self.args.train.slot is not None:
                        continue
                    labels[s][i][self.ontology.values[s].index(v)] = 1
            labels = {s: torch.Tensor(m).to(self.device) for s, m in labels.items()}

            loss = 0
            for s in self.ontology.slots:
                if s != self.args.train.slot and self.args.train.slot is not None:
                        continue
                loss += F.binary_cross_entropy(ys[s], labels[s])
        else:
            loss = torch.Tensor([0]).to(self.device)
        return loss, {s: v.data.tolist() for s, v in ys.items()}

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        summary = {}
        test_it, test_de = test
        #self.visualize()
        for set_name, dataset in {"train": train, "dev": dev, "test_it": test_it, "test_de": test_de}.items():
            if set_name == "test_it":
                self.ontology = self.ontology_tgt[0]
            elif set_name == "test_de":
                self.ontology = self.ontology_tgt[1]
            else:
                self.ontology = self.ontology_src
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))

    def start(self, inputs):
        train, dev, (test_it, test_de), _, _, _ = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, (test_it, test_de))
        if self.args.model.resume is not None:
            self.run_eval(train, dev, (test_it, test_de))
