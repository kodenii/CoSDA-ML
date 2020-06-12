import logging
import numpy as np
import random
import torch

import util.tool

from util.configue import Configure


def start():
    

    logging.basicConfig(level = logging.INFO)

    args = Configure.Get()

    np.random.seed(args.train.seed)
    random.seed(args.train.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.train.seed)
        torch.cuda.manual_seed(args.train.seed)

    torch.manual_seed(args.train.seed)
    torch.random.manual_seed(args.train.seed)


    Model, DatasetTool = util.tool.load_module(args.model.name, args.dataset.tool)

    inputs = DatasetTool.get(args)
    

    model = Model(args, DatasetTool, inputs)
    if args.train.gpu:
        model.cuda()

    model.start(inputs)

if __name__ == "__main__":
    start()
    