import csv
from pathlib import Path
import pathlib

class DictNormalizer(object):
    def __init__(self, dic):
        self.dic = dic

    def normalize(self, word):
        if word in self.dic:
            return self.dic[word]
        return ''


def load_dict(fn):
    if not isinstance(fn, pathlib.PurePath):
        fn = Path(fn)

    res = {}
    with open(str(fn), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            res[row[0]] = row[1]

    return res


