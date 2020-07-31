import argparse
import json
from pathlib import Path
import sys

from .ner import Ner
from .normalize import load_dict, DictNormalizer

parser = argparse.ArgumentParser()
parser.add_argument("-i", '--input_file', nargs='?', type=argparse.FileType(),
        default=sys.stdin, help="input file path")
parser.add_argument("-o", '--output_file', nargs='?', type=argparse.FileType(),
        default=sys.stdout, help="output file path")
parser.add_argument("-m", '--model_name', default='BERT', help="model file directory")
parser.add_argument("-n", '--normalizer', default='dict', help="dictionary file directory")
parser.add_argument("-f", '--output_format', default='xml', help="output format (xml or dict). default is xml")
args = parser.parse_args()

"""
assert args.input_file, 'Please specify input file name using -i option.'
fn = Path(args.input_file)
"""

form = args.output_format

"""
if args.output_file:
    ofn = Path(args.output_file)
else:
    ofn = Path('output.txt' if form == 'xml' else 'output.jsonl')
"""

sents = []
"""
with open(str(fn), 'r') as f:
"""
with args.input_file as f:
    for line in f:
        line = line.rstrip()
        if not line:
            continue
        sents.append(line)

model = Ner.from_pretrained(args.model_name, normalizer=args.normalizer)
results = model.predict(sents, form)

if form == 'xml':
    #with open(str(ofn), 'w') as f:
    with args.output_file as f:
        f.write('\n'.join(results))
else:
    outputs = []
    for r in results:
        outputs.append(json.dumps(r, ensure_ascii=False))
    print(outputs)
    #with open(str(ofn), 'w') as f:
    with args.output_file as f:
        f.write('\n'.join(outputs) + '\n')

