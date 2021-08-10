import unittest

from medner_j import Ner


class TestNer(unittest.TestCase):
    def setUp(self):
        self.model = Ner.from_pretrained(model_name="BERT", normalizer="dict")
        self.examples = ['それぞれの関節に関節液貯留は見られなかった', 'その後、左半身麻痺、ＣＴにて右前側頭葉の出血を認める。　']
        self.xmls = ['それぞれの関節に<CN value="かんせつえきちょりゅう;icd=E877;lv=C/freq=高;体液貯留">関節液貯留</CN>は見られなかった', 'その後、<C value="ひだりはんしんまひ;icd=G819;lv=A/freq=高;片麻痺">左半身麻痺</C>、ＣＴにて右前側頭葉の<C value="しゅっけつ;icd=R58;lv=S/freq=高;出血">出血</C>を認める。　']
        self.dicts = [
            [{"span": (8, 13), "type": "CN", "disease":"関節液貯留", "norm":"かんせつえきちょりゅう;icd=E877;lv=C/freq=高;体液貯留"}],
            [{"span": (4, 9), "type": "C", "disease": "左半身麻痺", "norm": "ひだりはんしんまひ;icd=G819;lv=A/freq=高;片麻痺"}, {"span": (20, 22), "type": "C", "disease": "出血", "norm": "しゅっけつ;icd=R58;lv=S/freq=高;出血"}]
        ]

    def test_xml(self):
        results = self.model.predict(self.examples)
        self.assertEqual(results, self.xmls)

    def test_dict(self):
        results = self.model.predict(self.examples, output_format="dict")
        self.assertEqual(results, self.dicts)

    def tearDown(self):
        del self.model
        del self.examples
        del self.xmls
        del self.dicts
