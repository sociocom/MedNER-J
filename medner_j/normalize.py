import csv
from pathlib import Path
import pathlib
import unicodedata

from .util import download_fileobj


def load_dict(fn):
    if not isinstance(fn, pathlib.PurePath):
        fn = Path(fn)

    res = {}
    with open(str(fn), 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            key = unicodedata.normalize('NFKC',row[0])
            res[key] = row[1]

    return res


class DictNormalizer(object):
    def __init__(self, fn):
        if not isinstance(fn, pathlib.PurePath):
            fn = Path(fn)

        if not fn.is_file():
            download_fileobj("http://aoi.naist.jp/MedEXJ2/norm_dic.csv", str(fn))

        self.dic = load_dict(fn)

    def normalize(self, word):
        word = unicodedata.normalize('NFKC', word)
        if word in self.dic:
            return self.dic[word]
        return ''


