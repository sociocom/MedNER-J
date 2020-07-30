import pathlib
from pathlib import Path
import itertools
import sys
import os

from transformers import BertJapaneseTokenizer, BertModel
from allennlp.modules.conditional_random_field import allowed_transitions
from model import BertCrf, ListTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

from util import create_label_vocab_from_file, convert_dict_to_xml, convert_iob_to_dict, download_fileobj
from normalize import load_dict, DictNormalizer

DEFAULT_CACHE_PATH = os.getenv("DEFAULT_CACHE_PATH", "~/.cache")
DEFAULT_MEDEXJ_PATH = Path(os.path.expanduser(
        os.path.join(DEFAULT_CACHE_PATH, "MedEXJ")
        ))
DEFAULT_MODEL_PATH = DEFAULT_MEDEXJ_PATH / "pretrained"

BERT_URL = "http://aoi.naist.jp/MedEXJ2/pretrained"


class Ner(object):
    def __init__(self, base_model, basic_tokenizer, subword_tokenizer, model_dir=DEFAULT_MODEL_PATH, normalizer=None):
        if not isinstance(model_dir, pathlib.PurePath):
            model_dir = Path(model_dir)

        label_vocab = create_label_vocab_from_file(str(model_dir / 'labels.txt'))
        self.itol = {i:l for l, i in label_vocab.items()}
        constraints = allowed_transitions("BIO", {i:w for w, i in label_vocab.items()})
        self.model = BertCrf(base_model, len(label_vocab), constraints)
        self.model.load_state_dict(torch.load(str(model_dir / 'final.model')))

        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)

        self.basic_tokenizer = basic_tokenizer
        self.subword_tokenizer = subword_tokenizer

        self.normalizer = normalizer


    def basic_tokenize(self, sents):
        return [self.basic_tokenizer.tokenize(s) for s in sents]

    def subword_tokenize(self, tokens):
        subwords = [[self.subword_tokenizer.tokenize(s) for s in ss] for ss in tokens]
        lengths = [[len(s) for s in ss] for ss in subwords]
        subwords = [list(itertools.chain.from_iterable(ss)) for ss in subwords]

        return subwords, lengths

    def numericalize(self, tokens):
        return [self.subword_tokenizer.convert_tokens_to_ids(t) for t in tokens]

    def encode(self, sents):
        tokens = self.basic_tokenize(sents)
        subwords, lengths = self.subword_tokenize(tokens)

        subwords = [["[CLS]"] + sub for sub in subwords]

        inputs = self.numericalize(subwords)
        inputs = [torch.tensor(i).to(self.device) for i in inputs]

        return inputs, lengths, tokens

    def _infer_space_tag(self, pre_tag, tag, post_tag):
        if pre_tag % 2 == 1 and post_tag == pre_tag + 1:
            return pre_tag + 1
        elif pre_tag != 0 and post_tag == pre_tag:
            return pre_tag
        return 0

    def integrate_subwords_tags(self, tags, lengths):
        # def merge(tags):
        #     return tags[0]

        results = []

        for ts, ls in zip(tags, lengths):
            result = []
            idx = 0
            for l in ls:
                if l == 0:
                    pre_tag = 0 if idx == 0 else ts[idx-1]
                    post_tag = 0 if idx == len(ts) - 1 else ts[idx+1]
                    tag = self._infer_space_tag(pre_tag, ts[idx], post_tag)
                else:
                    # tag = merge(ts[idx : idx + l])
                    tag = ts[idx : idx + l][0]
                idx += l
                result.append(tag)

            results.append(result)

        return results

    def predict(self, sents, output_format="xml"):
        inputs, lengths, tokens = self.encode(sents)
        results = []

        for s_idx in range(0, len(inputs) + 1, 16):
            e_idx = min(len(inputs), s_idx + 16)
            batch_inputs = inputs[s_idx:e_idx]
            padded_batch_inputs = pad_sequence(
                batch_inputs, batch_first=True, padding_value=0
            )
            mask = [[int(i > 0) for i in ii] for ii in padded_batch_inputs]
            mask = torch.tensor(mask).to(self.device)

            tags = self.model.decode(padded_batch_inputs, mask)
            tags = [t[0] for t in tags]
            tags = self.integrate_subwords_tags(tags, lengths[s_idx:e_idx])
            results.append(tags)

        results = [[self.itol[t] for t in tt] for tt in tags]

        results = convert_iob_to_dict(tokens, results)
        if self.normalizer is not None:
            self._normalize_from_dict(results)

        if output_format == "xml":
            results = convert_dict_to_xml(tokens, results)

        return results

    def _normalize_from_dict(self, dict_list):
        for dd in dict_list:
            for d in dd:
                d["norm"] = self.normalizer.normalize(d["disease"])

    @classmethod
    def from_pretrained(cls, model_name="BERT", normalizer='dict'):
        assert model_name == "BERT", "BERT以外未実装です"
        if model_name == "BERT":
            model_dir = DEFAULT_MODEL_PATH
            src_url = BERT_URL
            base_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-char')
            basic_tokenizer = ListTokenizer()
            subword_tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-char', do_basic_tokenize=False)

        if not model_dir.parent.is_dir():
            model_dir.parent.mkdir()

        if not model_dir.is_dir():
            model_dir.mkdir()

        if not (model_dir / "final.model").is_file():
            download_fileobj(src_url + "/final.model", model_dir / "final.model")
        if not (model_dir / "labels.txt").is_file():
            download_fileobj(src_url + "/labels.txt", model_dir / "labels.txt")

        if isinstance(normalizer, str):
            normalizer = DictNormalizer(DEFAULT_MEDEXJ_PATH / "norm_dic.csv")
        elif isinstance(normalizer, object):
            normalizer = normalizer
        else:
            raise TypeError

        ner = cls(base_model, basic_tokenizer, subword_tokenizer, model_dir=model_dir, normalizer=normalizer)

        return ner


if __name__ == "__main__":
    fn = Path(sys.argv[1])
    sents = []
    with open(str(fn), "r") as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            sents.append(line)

    print(sents)
    dic = load_dict("norm_dic.csv")
    normalizer = DictNormalizer(dic)
    model = Ner.from_pretrained(Path("pretrained"), normalizer=normalizer)
    results = model.predict(sents)
    print(results)
