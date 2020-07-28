import pathlib
from pathlib import Path
import itertools
import sys

from transformers import BertJapaneseTokenizer, BertModel
from allennlp.modules.conditional_random_field import allowed_transitions
from model import BertCrf
import torch
from torch.nn.utils.rnn import pad_sequence

from util import create_label_vocab_from_file, convert_dict_to_xml, convert_iob_to_dict
from normalize import load_dict, DictNormalizer

class Ner(object):
    def __init__(self, label_vocab, normalizer=None):
        self.itol = {i:l for l, i in label_vocab.items()}
        constraints = allowed_transitions("BIO", {i:w for w, i in label_vocab.items()})
        bert = BertModel.from_pretrained('bert-base-japanese-char')
        self.model = BertCrf(bert, len(label_vocab), constraints)

        self.tokenizer = BertJapaneseTokenizer.from_pretrained('bert-base-japanese-char', do_basic_tokenize=False)
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model.to(self.device)

        self.normalizer = normalizer

    def basic_tokenize(self, sents):
        return [list(s) for s in sents]

    def subword_tokenize(self, tokens):
        subwords = [[self.tokenizer.tokenize(s) for s in ss] for ss in tokens]
        lengths = [[len(s) for s in ss] for ss in subwords]
        subwords = [list(itertools.chain.from_iterable(ss)) for ss in subwords]

        return subwords, lengths

    def numericalize(self, tokens):
        return [self.tokenizer.convert_tokens_to_ids(t) for t in tokens]

    def encode(self, sents):
        tokens = self.basic_tokenize(sents)
        subwords, lengths = self.subword_tokenize(tokens)

        subwords = [['[CLS]'] + sub for sub in subwords]

        inputs = self.numericalize(subwords)
        inputs = [torch.tensor(i).to(self.device) for i in inputs]

        return inputs, lengths, tokens

    def integrate_subwords_tags(self, tags, lengths):
        def merge(tags):
            return tags[0]

        results = []

        for ts, ls in zip(tags, lengths):
            result = []
            idx = 0
            for l in ls:
                tag = merge(ts[idx:idx+l])
                idx += l
                result.append(tag)

            results.append(result)

        return results

    def predict(self, sents, output_format='xml'):
        inputs, lengths, tokens = self.encode(sents)
        results = []

        for s_idx in range(0, len(inputs)+1, 16):
            e_idx = min(len(inputs), s_idx+16)
            batch_inputs = inputs[s_idx:e_idx]
            padded_batch_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=0)
            mask = [[int(i>0) for i in ii] for ii in padded_batch_inputs]
            mask = torch.tensor(mask).to(self.device)

            tags = self.model.decode(padded_batch_inputs, mask)
            tags = [t[0] for t in tags]
            tags = self.integrate_subwords_tags(tags, lengths[s_idx:e_idx])
            results.append(tags)

        results = [[self.itol[t] for t in tt] for tt in tags]

        results = convert_iob_to_dict(tokens, results)
        if self.normalizer is not None:
            self._normalize_from_dict(results)

        if output_format == 'xml':
            results = convert_dict_to_xml(tokens, results)

        return results

    def _normalize_from_dict(self, dict_list):
        for dd in dict_list:
            for d in dd:
                d['norm'] = self.normalizer.normalize(d['disease'])


    @classmethod
    def from_pretrained(cls, fn, normalizer=None):
        if not isinstance(fn, pathlib.PurePath):
            fn = Path(fn)

        label_vocab = create_label_vocab_from_file(str(fn / 'labels.txt'))

        ner = cls(label_vocab, normalizer)
        ner.model.load_state_dict(torch.load(str(fn / 'final.model')))

        return ner

if __name__ == "__main__":
    fn = Path(sys.argv[1])
    sents = []
    with open(str(fn), 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            sents.append(line)

    print(sents)
    dic = load_dict('norm_dic.csv')
    normalizer = DictNormalizer(dic)
    model = Ner.from_pretrained(Path('pretrained'), normalizer=normalizer)
    results = model.predict(sents)
    print(results)
