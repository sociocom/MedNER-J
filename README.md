# MedNER-J
## 概要
日本語の病名抽出器である[MedEX/J](http://sociocom.jp/~data/2017-MEDEX/index.html)の最新バージョンです．

## 手法
Bidirectional Encoder Representations from Transformers (BERT)の特徴量を元に，条件付き確率場 (Conditional Random Fields: CRF) で病名の抽出を行っています．

BERTは[東北大学乾・鈴木研究室配布の文字ベースモデル](https://www.nlp.ecei.tohoku.ac.jp/news-release/3284/)を使用しています．

## インストール
python >= 3.6.1
```python setup.py install```

## 使い方
### コマンド
- -i：入力ファイル名
- -o：出力ファイル名
- -m：モデル（default: BERT）
- -n：正規化方法（default: dict）
- -f：出力フォーマット (xml or json, default:xml)


```python -m medner_j -i sample.txt -o output.txt -f xml```

入力ファイルは１行１文のテキストファイルを用意してください．

辞書は[万病辞書](http://sociocom.jp/~data/2018-manbyo/index.html)を使用しています．

xml形式とjson形式を選択できます．それぞれの出力フォーマットについては「使い方」の出力例をご参照ください．

（注）初回の動作時に，モデルファイルと辞書ファイルのダウンロードが行われます（~/.cache/MedNERJ）


### 使用例
#### 入力 (sample.txt)
```
それぞれの関節に関節液貯留は見られなかった
その後、左半身麻痺、ＣＴにて右前側頭葉の出血を認める。
```

#### コマンド
```python main.py -i sample.txt -o sample_output.txt -f xml```


#### 出力 (sample_output.txt) (xml形式)
```
それぞれの関節に<CN value="かんせつえきちょりゅう;icd=E877;lv=C/freq=高;体液貯留">関節液貯留</CN>は見られなかった
の後、<C value="ひだりはんしんまひ;icd=G819;lv=A/freq=高;片麻痺">左半身麻痺</C>、ＣＴにて右前側頭葉の<C value="しゅっけつ;icd=R58;lv=S/freq=高;出血">出血</C>を認める。
```

#### 出力 (sample_output.txt) (json形式)
```
[{"span": [8, 13], "type": "CN", "disease": "関節液貯留", "norm": "かんせつえきちょりゅう;icd=E877;lv=C/freq=高;体液貯留"}]
[{"span": [4, 9], "type": "C", "disease": "左半身麻痺", "norm": "ひだりはんしんまひ;icd=G819;lv=A/freq=高;片麻痺"}, {"span": [20, 22], "type": "C", "disease": "出血", "norm": "しゅっけつ;icd=R58;lv=S/freq=高;出血"}]
```

### スクリプト
```
from medner_j import Ner

sents = [
  "一方で、本邦では体交時の合併症への懸念、マンパワーの不足により腹臥位療法の実施が困難な場合が多い。",
  "腹臥位以外の時間は両側への完全側臥位を実施した。"
  ]
  
model = Ner.from_pretrained()
results = model.predict(sents)
print(results)
```
