# MedEXJ2
## 概要
日本語の病名抽出器である[MedEX/J](http://sociocom.jp/~data/2017-MEDEX/index.html)の最新バージョンです．

## 手法
Bidirectional Encoder Representations from Transformers (BERT)の特徴量を元に，条件付き確率場 (Conditional Random Fields: CRF) で病名の抽出を行っています．

BERTは[東北大学乾・鈴木研究室配布の文字ベースモデル](https://www.nlp.ecei.tohoku.ac.jp/news-release/3284/)を使用しています．

## requirements
- python 3.6.1 
- torch==1.4.0 
- transformers==2.8.0 
- allennlp==0.9.0 

一括インストールは以下のコマンドで行えます． 

```pip install -r requirements.txt```

## データ
- 学習済みモデルファイル
- 病名正規化用辞書ファイル

のダウンロードが必要です．以下のコマンドでダウンロードを行えます．

``` sh download_data.sh```

## 使い方
```python main.py -i input_file_path -o output_file_path -m model_dir -d dictionary_file_path -f output_format```

### 使用例
```python main.py -i sample.txt -o sample_output.txt -m data/pretrained -d data/norm_dic.txt -f xml```

- sample.txt
```
それぞれの関節に関節液貯留は見られなかった
その後、左半身麻痺、ＣＴにて右前側頭葉の出血を認める。
```

- sample_output.txt (xml形式)
```
それぞれの関節に<CN value="かんせつえきちょりゅう;icd=E877;lv=C/freq=高;体液貯留">関節液貯留</CN>は見られなかった
の後、<C value="ひだりはんしんまひ;icd=G819;lv=A/freq=高;片麻痺">左半身麻痺</C>、ＣＴにて右前側頭葉の<C value="しゅっけつ;icd=R58;lv=S/freq=高;出血">出血</C>を認める。
```

- sample_output.txt (json形式)
```
[{"span": [8, 13], "type": "CN", "disease": "関節液貯留", "norm": "かんせつえきちょりゅう;icd=E877;lv=C/freq=高;体液貯留"}]
[{"span": [4, 9], "type": "C", "disease": "左半身麻痺", "norm": "ひだりはんしんまひ;icd=G819;lv=A/freq=高;片麻痺"}, {"span": [20, 22], "type": "C", "disease": "出血", "norm": "しゅっけつ;icd=R58;lv=S/freq=高;出血"}]
```
