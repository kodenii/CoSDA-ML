import json

import util.tool

class Reader(object):
    def read_raw(file):
        with open(file, encoding="utf8") as reader:
            raw = reader.readlines()
        return raw

    def read_json(file):
        with open(file, encoding="utf8") as reader:
            raw = json.loads(reader.read())
        return raw

class Delexicalizer(object):
    def remove_linefeed(input):
        if isinstance(input, str):
            if input[-1] == "\n":
                return input[ : -1]
            else:
                return input
        elif isinstance(input, list):
            return util.tool.in_each(input, lambda x : Delexicalizer.remove_linefeed(x))
        else:
            return input

class Writer(object):
    def write_json(js, file, pretty = True):
        with open(file, "w", encoding="utf8") as writer:
            if pretty:
                writer.writelines(json.dumps(js, indent=4, separators=(',', ': ')))
            else:
                writer.writelines(json.dumps(js))
    def write_raw(raw, file):
        with open(file, "w", encoding="utf8") as writer:
            for line in raw:
                writer.writelines(line + "\n")