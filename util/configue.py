import configparser
import argparse
import os
import logging
import datetime

from util.tool import Args
from util.convert import String

DEFAULT_CONFIGURE_DIR = "configure"
DEFAULT_DATASET_DIR = "dataset"
DEFAULT_MODEL_DIR = "model"
DEFAULT_EXP_DIR = "exp"
DEFAULT_CONSOLE_ARGS_LABEL_FILE = "__console__.cfg"

class Configure(object):
    def get_cfg(file):
        cfgargs = Args()
        parser = configparser.ConfigParser()
        parser.read(file)
        for section in parser.sections():
            setattr(cfgargs, section, Args())
            for item in parser.items(section):
                setattr(getattr(cfgargs, section), item[0], String.to_basic(item[1]))
        return cfgargs

    def get_con(default_file):
        conargs = Args()
        parser = argparse.ArgumentParser()
        types = {"bool": bool, "int": int, "float": float}
        args_label = Configure.get_cfg(default_file)
        for arg_name, arg in args_label:
            argw = {}
            if arg.help:
                argw["help"] = arg.help
            if arg.type == "implicit_bool" or arg.type == "imp_bool":
                argw["action"] = "store_true"
            if arg.type == "string" or arg.type == "str" or arg.type is None:
                if arg.default:
                    if arg.default == "None" or "none":
                        argw["default"] = None
                    else:
                        argw["default"] = arg.default
            if arg.type in types:
                argw["type"] = types[arg.type]
                if arg.default:
                    if arg.default == "None" or "none":
                        argw["default"] = None
                    else:
                        argw["default"] = types[arg.type](arg.default)
            parser.add_argument("--" + arg_name, **argw)
        tmpargs = parser.parse_args()
        for arg_name, arg in args_label:
            setattr(conargs, arg_name, getattr(tmpargs, arg_name))
        return conargs

    def Get():
        conargs = Configure.get_con(os.path.join(DEFAULT_CONFIGURE_DIR, DEFAULT_CONSOLE_ARGS_LABEL_FILE))
        logging.info("Loading configure from " + conargs.cfg)
        args = Configure.get_cfg(os.path.join(DEFAULT_CONFIGURE_DIR, conargs.cfg))
        if conargs.debug:
            logging.debug("Debug flag found")
            for arg_name, arg in args.debug:
                cur = args
                arg_divs = arg_name.split(".")
                for arg_div in arg_divs[ : -1]:
                    cur = getattr(cur, arg_div)
                setattr(cur, arg_divs[-1], arg)
                delattr(args.debug, arg_name)
            args.debug = True
        if not args.model.nick:
            args.model.nick = args.model.name + "," + str(datetime.datetime.now()).replace(":", ".").replace(" ", ",")[0:19]
        if args.dir is not Args:
            args.dir = Args()
        args.dir.model = DEFAULT_MODEL_DIR
        args.dir.exp = DEFAULT_EXP_DIR
        args.dir.dataset = DEFAULT_DATASET_DIR
        args.dir.configure = DEFAULT_CONFIGURE_DIR
        args.dir.output = os.path.join(args.dir.exp, args.model.nick)
        for arg_name, arg in conargs:
            if arg is None:
                continue
            if arg_name != "cfg":
                names = arg_name.split(".")
                cur = args
                for name in names[ : -1]:
                    if getattr(cur, name) is None:
                        setattr(cur, name, Args())
                    cur = getattr(cur, name)
                setattr(cur, names[-1], arg)
        return args