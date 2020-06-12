import torch

import util.tool

class String(object):
    def to_basic(string):
        try:
            return int(string)
        except:
            pass
        try:
            return float(string)
        except:
            pass
        if string in ["True", "true"]:
            return True
        elif string in ["False", "false"]:
            return False
        else:
            return string

class Common(object):
    def to_args(src, recursive = False):
        if not isinstance(src, dict) and not isinstance(src, list):
            return src

        if isinstance(src, dict):
            tgt = util.tool.Args()
            for key in src:
                if recursive:
                    setattr(tgt, key, Common.to_args(src[key], recursive))
                else:
                    setattr(tgt, key, src[key])
        else:
            if recursive:
                tgt = util.tool.in_each(src, lambda x : Common.to_args(x, recursive))
            else:
                tgt = src
        return tgt

class List(object):
    def to_bert_msk_and_idx(src, source_len, max_len, bias):
        idx = util.tool.in_each(src, lambda x : [(1, 1)] + x[1 :])
        idx = util.tool.in_each(idx, lambda x : util.tool.idx_extender(x, max_len, 0, bias = bias))
        msk = util.tool.in_each(source_len, lambda x : [1] * x + [0] * (max_len - x))
        return idx, msk

    def to_str(src):
        if isinstance(src, str):
            return src
        out = ""
        for ele in src:
            out += str(ele)
        return out

    def to_bert_token_idx(src, tokener, max_len = 256):
        x = tokener.convert_tokens_to_ids(tokener.tokenize(util.convert.List.to_str(src)))
        # space maybe convert into anything
        if src == " ":
            x = [100]
        return x[:min(len(x),max_len)]

    def to_bert_info(inputs, tokener, pad, cls, device, max_len = 256):
        raw_source = util.tool.in_each(inputs, lambda x : util.convert.List.to_bert_token_idx(x, tokener, max_len))
        source_len = util.tool.in_each(raw_source, lambda x : len(x) + 1)
        source, pad_idx = util.tool.pad([[[cls]] * len(inputs), raw_source], pad)
        if source == []:
            mmax_len = 0
        else:
            mmax_len = len(source[0])
        source_idx, source_msk = util.convert.List.to_bert_msk_and_idx(pad_idx, source_len, mmax_len, -1)
        return (torch.Tensor(source).long().to(device), torch.Tensor(source_idx).long().to(device), torch.Tensor(source_msk).long().to(device)), source_len

    def to_xlm_info(inputs, tokener, pad, cls, device, max_len = 256):
        raw_source = util.tool.in_each(inputs, lambda x : util.convert.List.to_bert_token_idx(x, tokener, max_len))
        source_len = util.tool.in_each(raw_source, lambda x : len(x) + 1)
        source, pad_idx = util.tool.pad([[[cls]] * len(inputs), raw_source], pad)
        if source == []:
            mmax_len = 0
        else:
            mmax_len = len(source[0])
        source_idx, source_msk = util.convert.List.to_bert_msk_and_idx(pad_idx, source_len, mmax_len, -1)
        return torch.Tensor(source).long().to(device), torch.Tensor(source_msk).long().to(device)

    def to_bert_info2(inputs1, inputs2, tokener, pad, cls, sep, device, max_len = 256):
        raw_source1 = util.tool.in_each(inputs1, lambda x : util.convert.List.to_bert_token_idx(x, tokener, max_len))
        raw_source2 = util.tool.in_each(inputs2, lambda x : util.convert.List.to_bert_token_idx(x, tokener, max_len))
        raw_source = util.tool.in_each(zip(raw_source1, raw_source2), lambda x : x[0] + [sep] + x[1] + [sep])
        source_len = util.tool.in_each(raw_source, lambda x : len(x) + 1)
        source, pad_idx = util.tool.pad([[[cls]] * len(raw_source), raw_source], pad)
        if source == []:
            max_len = 0
        else:
            max_len = len(source[0])
        source_idx, source_msk = util.convert.List.to_bert_msk_and_idx(pad_idx, source_len, max_len, -1)
        return (torch.Tensor(source).long().to(device), torch.Tensor(source_idx).long().to(device), torch.Tensor(source_msk).long().to(device)), source_len