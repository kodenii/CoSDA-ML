import os

import random
import numpy as np

import util.data
import util.convert
import util.tool
from pytorch_transformers import BertTokenizer

class DatasetTool(object):

    def get_emb(file):
        return util.data.Reader.read_json(file)

    def trans(x):
        fix = [("food", "cibo", "essen"), ("price", "prezzo", "preisklasse"), ("area", "area", "gegend")]
        for k in fix:
            if x in k:
                return k[0]
        return x

    def get_ontology(file, dontcare):
        raw = util.data.Reader.read_json(file)
        ontology = {"slots": [], "values": {}, "num": {}}
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number', "price range": "price"}
        for slot in raw["informable"]:
            if slot in fix:
                raw["informable"][fix[slot]] = raw["informable"][slot]
                del raw["informable"][slot]
                slot = fix[slot]
            ontology["slots"].append(DatasetTool.trans(slot))
            ontology["values"][DatasetTool.trans(slot)] = []
            for value in raw["informable"][slot]:
                if value in fix:
                    value = fix[value]
                ontology["values"][DatasetTool.trans(slot)].append(value)
            if slot != "request":
                ontology["values"][DatasetTool.trans(slot)].append(dontcare)
        ontology = util.convert.Common.to_args(ontology)
        return ontology

    def get_set(file, trans = False, idx_dict = None):
        raw = util.data.Reader.read_json(file)
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number', "price range": "price"}
        turns = []
        for dia in raw:
            for turn in dia["dialogue"]:
                turn["dialogue_id"] = dia["dialogue_idx"]
                for x in turn["turn_label"]:
                    if x[0] in fix:
                        x[0] = fix[x[0]]
                    if x[1] in fix:
                        x[1] = fix[x[1]]
                if not trans:
                    turn["system_acts"] = ["request " + x if isinstance(x, str) else \
                        "inform " + x[0] + " = " + x[1] for x in turn["system_acts"]]
                    turn["transcript"] = turn["transcript"].lower()
                else:
                    turn["system_acts"] = [idx_dict.src2tgt["request"][-1] + " " + x if isinstance(x, str) else \
                        idx_dict.src2tgt["inform"][-1] + " " + x[0] + " = " + x[1] for x in turn["system_acts"]]
                turn["system_acts"].append("")
                turns.append(turn)
        dataset = util.convert.Common.to_args(turns, True)
        return dataset

    def get_idx_dict(file, args, ontology_src, ontology_tgt):
        raw = util.data.Delexicalizer.remove_linefeed(util.data.Reader.read_raw(file))
        idx_dict = util.convert.Common.to_args({"src2tgt": {}, "tgt2src": {}})
        if args.dataset.part is not None:
            raw = raw[ : args.dataset.part * 100]
        for line in raw:
            try:
                src, tgt = line.split("\t")
            except:
                src, tgt = line.split(" ")
            if src not in idx_dict.src2tgt:
                idx_dict.src2tgt[src] = [tgt]
            else:
                idx_dict.src2tgt[src].append(tgt)
        for slot in ontology_src.slots:
            for (x, y) in zip(ontology_src.values[slot], ontology_tgt.values[slot]):
                if x not in idx_dict.src2tgt:
                    idx_dict.src2tgt[x] = [y]
                else:
                    idx_dict.src2tgt[x].append(y)
        return idx_dict

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
        train_file = os.path.join(args.dir.dataset, args.dataset.src, "train.json")
        dev_file = os.path.join(args.dir.dataset, args.dataset.src, "dev.json")
        test_file = os.path.join(args.dir.dataset, args.dataset.tgt, "test.json")
        ontology_src_file = os.path.join(args.dir.dataset, args.dataset.src, "ontology.json")
        ontology_tgt_file = os.path.join(args.dir.dataset, args.dataset.tgt, "ontology.json")
        dict_file = os.path.join(args.dir.dataset, args.dataset.dict)
        train = DatasetTool.get_set(train_file)
        dev = DatasetTool.get_set(dev_file)
        ontology_src = DatasetTool.get_ontology(ontology_src_file, args.dataset.dontcare_src)
        ontology_tgt = DatasetTool.get_ontology(ontology_tgt_file, args.dataset.dontcare_tgt)
        word_dict = DatasetTool.get_idx_dict(dict_file, args, ontology_src, ontology_tgt)
        test = DatasetTool.get_set(test_file, True, word_dict)
        return train, dev, test, (ontology_src, ontology_tgt), word_dict, None

    def evaluate(pred, dataset, args):
        dialogue_id = -1
        request_score = []
        inform_score = []
        area_score = []
        food_score = []
        price_score = []
        jnt_score = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number', "price range": "price"}
        area_trans = ["area", "area", "gegend"]
        price_trans = ["price", "prezzo", "preisklasse"]
        food_trans = ["food", "cibo", "essen"]
        for turn_info, turn_pred in zip(dataset, pred):
            if turn_info.dialogue_id != dialogue_id:
                dialogue_id = turn_info.dialogue_id
                pred_state = {}
            request_goal = set([('request', v) for s, v in turn_info.turn_label if s == 'request'])
            area_goal = set([("area", v) for s, v in turn_info.turn_label if s in area_trans])
            food_goal = set([("food", v) for s, v in turn_info.turn_label if s in food_trans])
            price_goal = set([("price", v) for s, v in turn_info.turn_label if s in price_trans])
            inform_goal = area_goal | food_goal | price_goal
            request_pred = set([('request', v) for s, v in turn_pred if s == 'request'])
            area_pred = set([("area", v) for s, v in turn_pred if s in area_trans])
            food_pred = set([("food", v) for s, v in turn_pred if s in food_trans])
            price_pred = set([("price", v) for s, v in turn_pred if s in price_trans])
            inform_pred = area_pred | food_pred | price_pred
            request_score.append(request_goal == request_pred)
            inform_score.append(inform_goal == inform_pred)
            area_score.append(area_goal == area_pred)
            food_score.append(food_goal == food_pred)
            price_score.append(price_goal == price_pred)

            gold_recovered = set()
            pred_recovered = set()
            for s, v in inform_pred:
                pred_state[s] = v
            for b in turn_info.belief_state:
                for s, v in b.slots:
                    if b.act != 'request':
                        if fix.get(s.strip(), s.strip()) in area_trans:
                            gold_recovered.add((b.act, "area", fix.get(v.strip(), v.strip())))
                        if fix.get(s.strip(), s.strip()) in price_trans:
                            gold_recovered.add((b.act, "price", fix.get(v.strip(), v.strip())))
                        if fix.get(s.strip(), s.strip()) in food_trans:
                            gold_recovered.add((b.act, "food", fix.get(v.strip(), v.strip())))
            for s, v in pred_state.items():
                pred_recovered.add(('inform', s, v))
            turn_label = {}
            jnt_score.append(gold_recovered == pred_recovered)
        return {'turn_inform': np.mean(inform_score),"turn_area":np.mean(area_score),"turn_food":np.mean(food_score),"turn_price":np.mean(price_score), 'turn_request': np.mean(request_score), 'joint_goal': np.mean(jnt_score)}

    def record(pred, dataset, set_name, args):
        dialogue_id = -1
        js = []
        fix = {'centre': 'center', 'areas': 'area', 'phone number': 'number', "price range": "price"}
        area_trans = ["area", "area", "gegend"]
        price_trans = ["price", "prezzo", "preisklasse"]
        food_trans = ["food", "cibo", "essen"]
        for turn_info, turn_pred in zip(dataset, pred):
            if turn_info.dialogue_id != dialogue_id:
                dialogue_id = turn_info.dialogue_id
                pred_state = {}
            js_turn = {}
            js_turn["turn_pred"] = []
            js_turn["turn_goal"] = []
            request_goal = set([('request', v) for s, v in turn_info.turn_label if s == 'request'])
            area_goal = set([("area", v) for s, v in turn_info.turn_label if s in area_trans])
            food_goal = set([("food", v) for s, v in turn_info.turn_label if s in food_trans])
            price_goal = set([("price", v) for s, v in turn_info.turn_label if s in price_trans])
            inform_goal = area_goal | food_goal | price_goal
            js_turn["turn_goal"].append(list(area_goal))
            js_turn["turn_goal"].append(list(food_goal))
            js_turn["turn_goal"].append(list(price_goal))
            js_turn["turn_goal"].append(list(request_goal))

            request_pred = set([('request', v) for s, v in turn_pred if s == 'request'])
            area_pred = set([("area", v) for s, v in turn_pred if s in area_trans])
            food_pred = set([("food", v) for s, v in turn_pred if s in food_trans])
            price_pred = set([("price", v) for s, v in turn_pred if s in price_trans])
            inform_pred = area_pred | food_pred | price_pred
            js_turn["turn_pred"].append(list(area_pred))
            js_turn["turn_pred"].append(list(food_pred))
            js_turn["turn_pred"].append(list(price_pred))
            js_turn["turn_pred"].append(list(request_pred))

            gold_recovered = set()
            pred_recovered = set()
            for s, v in inform_pred:
                pred_state[s] = v
            for b in turn_info.belief_state:
                for s, v in b.slots:
                    if b.act != 'request':
                        gold_recovered.add((b.act, fix.get(s.strip(), s.strip()), fix.get(v.strip(), v.strip())))
            for s, v in pred_state.items():
                pred_recovered.add(('inform', s, v))
            js_turn["joint_goal"] = list(gold_recovered)
            js_turn["joint_pred"] = list(pred_recovered)
            js.append(js_turn)
        if not os.path.exists(args.dir.output):
            os.makedirs(args.dir.output)
        util.data.Writer.write_json(js, "{}/{}.json".format(args.dir.output, set_name))