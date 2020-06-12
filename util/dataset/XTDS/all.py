import os

import numpy as np
import random

import util.data
import util.convert
import util.tool

class DatasetTool(object):

    

    def get_set(file, ontology):
        with open(file, encoding="utf8") as reader:
            groups = "".join(reader.readlines()).split("\n\n")
        
        dataset = []
        for group in groups:
            if group == "":
                continue
            
            lines = group.split("\n")
            intent = lines[1][10:]
            
            if intent not in ontology["intent"]:
                ontology["intent"].append(intent)

            perdata = {"intent": ontology["intent"].index(intent), 
                       "utterance": [],
                       "slot": []}

            for line in lines[3:]:
                line = line.strip()
                ids, token, domain, slot = line.split("\t")
                if slot not in ontology["slot"]:
                    ontology["slot"].append(slot)
                perdata["slot"].append(ontology["slot"].index(slot))
                perdata["utterance"].append(token)
                
            dataset.append(perdata)
        return dataset

    def get_idx_dict(idx_dict, file, args):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        if args.train.dict_size is not None:
            raw = raw[:int(len(raw) * args.train.dict_size)]
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
        ontology = {"slot": [], "intent": []}
        train_file = os.path.join(args.dir.dataset, args.dataset.train)
        dev_file = os.path.join(args.dir.dataset, args.dataset.dev)
        args.dict_list = args.dataset.dict.split(" ")
        args.test_list = args.dataset.test.split(" ")
        train = DatasetTool.get_set(train_file, ontology)
        random.shuffle(train)
        dev = DatasetTool.get_set(dev_file, ontology)
        test = {test: DatasetTool.get_set(os.path.join(args.dir.dataset, test), ontology) for test in args.test_list}
        args.ontology = ontology
        idx_dict = util.convert.Common.to_args({"src2tgt": []})
        for dict_file in args.dict_list:
            dict_file = os.path.join(args.dir.dataset, dict_file)
            DatasetTool.get_idx_dict(idx_dict, dict_file, args)
        if args.train.train_size is not None:
            train = train[:int(len(train) * args.train.train_size)]
        return train, dev, test, ontology, idx_dict, None

    def evaluate(pred, dataset, args):
        summary = {}
        i_correct = 0
        d_correct = 0
        all = 0
        if not os.path.exists(args.dir.output):
            os.makedirs(args.dir.output)
        with open(os.path.join(args.dir.output, "eval.txt"), "w", encoding="utf8") as writer:
            for p, g in zip(pred, dataset):
                all += 1
                if p[0] == g["intent"]:
                    i_correct += 1
                for pp, gg in zip(p[1], g["slot"]):
                    goal_l = args.ontology["slot"][gg]
                    pred_l = args.ontology["slot"][pp]
                    if goal_l == "NoLabel":
                        goal_l = "O"
                    if pred_l == "NoLabel":
                        pred_l = "O"
                    writer.writelines("{}\t{}\t{}\t{}\t{}\n".format("x", "n", "O", goal_l, pred_l))
        out = os.popen('perl ./tool/conlleval.pl -d \"\\t\" < {}'.format(os.path.join(args.dir.output, "eval.txt"))).readlines()
        summary.update({"slot_f1": float(out[1][out[1].find("FB1:") + 4:-1].replace(" ", "")) / 100})
        summary.update({"intent_accuracy": i_correct / all})
        return summary

    def record(pred, dataset, set_name, args):
        pass
