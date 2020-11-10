"""日本語医療文書のための病名抽出システム

症例報告などの医療文書から病名を抽出（・正規化）するシステムです．
BERT-CRFを使用しています

Args:
    DEFAULT_CACHE_PATH (env): モデルのダウンロード先指定のための環境変数（default: ~/.cache）
"""

import pathlib
from pathlib import Path
import itertools
import sys
import os
from logging import getLogger, StreamHandler, INFO

from transformers import BertJapaneseTokenizer, BertModel
from allennlp.modules.conditional_random_field import allowed_transitions
import torch
from torch.nn.utils.rnn import pad_sequence

from .model import BertCrf, ListTokenizer
from .util import (
    create_label_vocab_from_file,
    convert_dict_to_xml,
    convert_iob_to_dict,
    download_fileobj,
)
from .normalize import load_dict, DictNormalizer


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False

DEFAULT_CACHE_PATH = os.getenv("DEFAULT_CACHE_PATH", "~/.cache")
DEFAULT_MEDNERJ_PATH = Path(
    os.path.expanduser(os.path.join(DEFAULT_CACHE_PATH, "MedNERJ"))
)
DEFAULT_MODEL_PATH = DEFAULT_MEDNERJ_PATH / "pretrained"
RADIOLOGY_MODEL_PATH = DEFAULT_MEDNERJ_PATH / "radiology"

BERT_URL = "http://aoi.naist.jp/MedEXJ2/pretrained"


class Ner(object):
    """NER model

    本体のモデルです．
    基本的に，from_pretrained()を使用してインスタンスを生成してください．

    Examples:
        インスタンスの生成::

            from medner_j import Ner
            model = Ner.from_pretrained()

    Args:
        label_vocab (dict): {label:label_idx, ...}
        itol (dict): {label_idx: label, ...}
        basic_tokenizer (callable): 単語分割用トークナイザ
        subword_tokenizer (callable): サブワード分割用トークナイザ
        model (nn.Module): BertCrfモデル
        normalizer (callable): 単語正規化関数
    """

    def __init__(
        self,
        base_model,
        basic_tokenizer,
        subword_tokenizer,
        model_dir=DEFAULT_MODEL_PATH,
        normalizer=None,
    ):
        """初期化

        非推奨です

        Args:
            base_model (nn.Module): BertCrfモデル
            basic_tokenizer (callable): 単語分割用トークナイザ
            subword_tokenizer (callable): サブワード分割用トークナイザ
            label_vocab (dict): {label:label_idx, ...}
            model_dir (pathlib.Path or str): モデルフォルダのpath．labels.txtとfinal.model
            normalizer (callable): 単語正規化関数
        """
        if not isinstance(model_dir, pathlib.PurePath):
            model_dir = Path(model_dir)

        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = "cpu"

        label_vocab = create_label_vocab_from_file(str(model_dir / "labels.txt"))
        self.itol = {i: l for l, i in label_vocab.items()}
        constraints = allowed_transitions("BIO", {i: w for w, i in label_vocab.items()})
        self.model = BertCrf(base_model, len(label_vocab), constraints)
        self.model.load_state_dict(
            torch.load(str(model_dir / "final.model"), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

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
        results = []

        for ts, ls in zip(tags, lengths):
            result = []
            idx = 0
            for l in ls:
                if l == 0:
                    pre_tag = 0 if idx == 0 else ts[idx - 1]
                    post_tag = 0 if idx == len(ts) - 1 else ts[idx + 1]
                    tag = self._infer_space_tag(pre_tag, ts[idx], post_tag)
                else:
                    # tag = merge(ts[idx : idx + l])
                    tag = ts[idx : idx + l][0]
                idx += l
                result.append(tag)

            results.append(result)

        return results

    def predict(self, sents, output_format="xml"):
        """病名抽出

        文のリストを受け取り，病名を抽出するメソッド

        Args:
            sents (List): 入力文のリスト
            output_format (str): 出力フォーマット．xml or dict（default: xml）

        Returns:
            List: 出力のリスト

        出力フォーマット（xml）::

            ["<C>脳梗塞</C>を認める．"]

        出力フォーマット（dict）::

            [{"span": (0, 3), "type": "C", "disease":"脳梗塞", "norm":"脳梗塞"}]

        """
        inputs, lengths, tokens = self.encode(sents)
        results = []

        for s_idx in range(0, len(inputs), 16):
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
            results.extend(tags)

        results = [[self.itol[t] for t in tt] for tt in results]

        results = convert_iob_to_dict(tokens, results)
        if self.normalizer is not None:
            self._normalize(results)

        if output_format == "xml":
            results = convert_dict_to_xml(tokens, results)

        return results

    def _normalize(self, dict_list):
        for dd in dict_list:
            for d in dd:
                d["norm"] = self.normalizer(d["disease"])

    @classmethod
    def from_pretrained(cls, model_name="BERT", normalizer="dict"):
        """学習モデルの読み込み

        学習済みモデルを読み込み，Nerインスタンスを返します．
        学習済みモデルがキャッシュされていない場合，~/.cacheにモデルのダウンロードを行います．
        ダウンロード先を指定したい場合は環境変数DEFAULT_CACHE_PATHで指定してください．

        Args:
            model_name (str): モデル名．現バージョンはBERTのみしか実装していません．
            normalizer (str or callable): 標準化方法の指定．dict or dnorm．

        Returns:
            Ner: Nerインスタンス
        """

        assert model_name in ["BERT", "radiology"], "BERT以外未実装です"
        if model_name in ["BERT", "radiology"]:
            model_dir = DEFAULT_MODEL_PATH
            if model_name == "radiology":
                model_dir = RADIOLOGY_MODEL_PATH
            src_url = BERT_URL
            base_model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese-char")
            basic_tokenizer = ListTokenizer()
            subword_tokenizer = BertJapaneseTokenizer.from_pretrained(
                "cl-tohoku/bert-base-japanese-char", do_word_tokenize=False
            )

        if not model_dir.parent.is_dir():
            logger.info("creating %s", str(model_dir.parent))
            model_dir.parent.mkdir()

        if not model_dir.is_dir():
            logger.info("creating %s", str(model_dir))
            model_dir.mkdir()

        if not (model_dir / "final.model").is_file():
            logger.info("not found %s", str(model_dir / "final.model"))
            download_fileobj(src_url + "/final.model", model_dir / "final.model")
        if not (model_dir / "labels.txt").is_file():
            logger.info("not found %s", str(model_dir / "labels.txt"))
            download_fileobj(src_url + "/labels.txt", model_dir / "labels.txt")

        if isinstance(normalizer, str):
            if normalizer == "dnorm":
                logger.info("try %s normalizer", "dnorm")
                try:
                    from dnorm_j import DNorm

                    normalizer = DNorm.from_pretrained().normalize
                    logger.info("use %s normalizer", "dnorm")
                except:
                    logger.warning("You did not install dnorm")
                    logger.warning("use %s normalizer", "Dict")
                    normalizer = DictNormalizer(
                        DEFAULT_MEDNERJ_PATH / "norm_dic.csv"
                    ).normalize
            else:
                logger.info("use %s normalizer", "Dict")
                normalizer = DictNormalizer(
                    DEFAULT_MEDNERJ_PATH / "norm_dic.csv"
                ).normalize

        elif isinstance(normalizer, object):
            logger.info("use %s normalizer", "your original")
            normalizer = normalizer
        else:
            raise TypeError

        ner = cls(
            base_model,
            basic_tokenizer,
            subword_tokenizer,
            model_dir=model_dir,
            normalizer=normalizer,
        )

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
    model = Ner.from_pretrained()
    results = model.predict(sents)
    print(results)
