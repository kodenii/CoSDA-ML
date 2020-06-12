import logging
import torch
import pprint
import re
import os

from tqdm import tqdm

import util.tool

from util.tool import Batch

class Model(torch.nn.Module):
    def __init__(self, args, DatasetTool, inputs):
        super().__init__()
        self.args = args
        self.optimizer = None
        self.DatasetTool = DatasetTool
    
    @property
    def device(self):
        if self.args.train.gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def forward(self, batch):
        raise NotImplementedError

    def set_optimizer(self):
        all_params = set(self.parameters())
        params = [{"params": list(all_params), "lr": self.args.lr.default}]
        self.optimizer = torch.optim.Adam(params)

    def get_pred(self, out):
        raise NotImplementedError

    def get_max_train(self, dataset):
        if self.args.dataset.part:
            max_train = min(self.args.dataset.part, len(dataset))
        else:
            max_train = len(dataset)
        return max_train

    def run_test(self, dataset):
        self.eval()
        all_out = []
        for batch in tqdm(Batch.to_list(dataset, self.args.train.batch)[0 : self.get_max_train(dataset)]):
            loss, out = self.forward(batch)
            all_out += self.get_pred(out)
        return self.DatasetTool.evaluate(all_out, dataset, self.args), all_out

    def run_batches(self, dataset, epoch):
        all_loss = 0
        all_size = 0
        iteration = 0
        for batch in tqdm(Batch.to_list(dataset, self.args.train.batch)[0 : self.get_max_train(dataset)]):
            loss, _ = self.forward(batch)
            self.zero_grad()
            loss.backward()
            self.optimizer.step()
            all_loss += loss.item()
            iteration += 1
            if self.args.train.iter_save is not None:
                if iteration % self.args.train.iter_save == 0:
                    if self.args.train.max_save > 0:
                        self.save('epoch={epoch},iter={iter}'.format(epoch = epoch, iter = iteration))
                        self.clear_saves()
            all_size += len(batch)
        return all_loss / all_size, iteration

    def update_best(self, best, summary, epoch):
        stop_key = 'eval_dev_{}'.format(self.args.train.stop)
        train_key = 'eval_train_{}'.format(self.args.train.stop)
        if self.args.train.not_eval or (best.get(stop_key, 0) <= summary[stop_key] and self.args.train.stopmin is None) or (best.get(stop_key, summary[stop_key]) >= summary[stop_key] and self.args.train.stopmin is not None):
            if self.args.train.not_eval:
                best.update(summary)
                if self.args.train.max_save > 0:
                    self.save('epoch={epoch}'.format(epoch = epoch))
                    self.clear_saves()
            else:
                best_dev = '{:f}'.format(summary[stop_key])
                best_train = '{:f}'.format(summary[train_key])
                best.update(summary)
                if self.args.train.max_save > 0:
                    self.save('epoch={epoch},train_{key}={train},dev_{key}={dev}'.format(epoch = epoch, train = best_train, dev = best_dev, key = self.args.train.stop))
                    self.clear_saves()
        return best

    def get_summary(self, epoch, iteration):
        return {"epoch": epoch, "iteration": iteration}

    def update_summary(self, summary, tmp_summary):
        pass

    def run_train(self, train, dev, test):
        self.set_optimizer()
        iteration = 0
        best = {}
        for epoch in range(self.args.train.epoch):
            self.train()
            logging.info("Starting training epoch {}".format(epoch))
            summary = self.get_summary(epoch, iteration)
            loss, iter = self.run_batches(train, epoch)
            iteration += iter
            summary.update({"loss": loss})
            if not self.args.train.not_eval:
                for set_name, dataset in {"train": train, "dev": dev, "test": test}.items():
                    tmp_summary, pred = self.run_test(dataset)
                    self.DatasetTool.record(pred, dataset, set_name, self.args)
                    summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
            best = self.update_best(best, summary, epoch)
            logging.info(pprint.pformat(best))
            logging.info(pprint.pformat(summary))

    def run_eval(self, train, dev, test):
        logging.info("Starting evaluation")
        summary = {}
        for set_name, dataset in {"train": train, "dev": dev, "test": test}.items():
            tmp_summary, pred = self.run_test(dataset)
            self.DatasetTool.record(pred, dataset, set_name, self.args)
            summary.update({"eval_{}_{}".format(set_name, k): v for k, v in tmp_summary.items()})
        logging.info(pprint.pformat(summary))
           

    def start(self, inputs):
        train, dev, test = inputs
        if self.args.model.resume is not None:
            self.load(self.args.model.resume)
        if not self.args.model.test:
            self.run_train(train, dev, test)
        if self.args.model.resume is not None:
            self.run_eval(train, dev, test)

    def load(self, file):
        logging.info("Loading model from {}".format(file))
        state = torch.load(file)
        model_state = state["model"]
        self.load_state_dict(model_state)

    def save(self, name):
        file = "{}/{}.pkl".format(self.args.dir.output, name)
        if not os.path.exists(self.args.dir.output):
            os.makedirs(self.args.dir.output)
        logging.info("Saving model to {}".format(name))
        state = {
            "model": self.state_dict()
        }
        torch.save(state, file)

    def get_saves(self):
        files = [f for f in os.listdir(self.args.dir.output) if f.endswith('.pkl')]
        scores = []
        for name in files:
            re_str = r'dev_{}=([0-9\.]+)'.format(self.args.train.stop)
            dev_acc = re.findall(re_str, name)
            if dev_acc:
                score = float(dev_acc[0].strip('.'))
                scores.append((score, os.path.join(self.args.dir.output, name)))
        scores.sort(key=lambda tup: tup[0], reverse=True)
        return scores

    def clear_saves(self):
        scores_and_files = self.get_saves()
        if len(scores_and_files) > self.args.train.max_save:
            for score, name in scores_and_files[self.args.train.max_save : ]:
                os.remove(name)