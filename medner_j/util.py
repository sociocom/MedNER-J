import requests
from logging import getLogger, StreamHandler, INFO

from tqdm import tqdm


logger = getLogger(__name__)
handler = StreamHandler()
handler.setLevel(INFO)
logger.setLevel(INFO)
logger.addHandler(handler)
logger.propagate = False


def download_fileobj(src, dst):
    logger.info("Downloading %s to %s", src, dst)

    file_size = int(requests.head(src).headers["content-length"])
    logger.info("File size of %s: %s KB", src, str(file_size // 1024))

    res = requests.get(src, stream=True)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True)

    with open(dst, "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            f.write(chunk)
            pbar.update(len(chunk))
        pbar.close()
        #shutil.copyfileobj(res.raw, f)

    logger.info("Finish downloading %s to %s", src, dst)


def convert_iob_to_dict(tokens, iobs):
    results = []
    for tt, ii in zip(tokens, iobs):
        assert len(tt) == len(ii), ''

        ii = ['O'] + ii + ['O']
        s_pos = -1
        word = ''
        result = []
        for idx in range(1, len(ii)-1):
            prefix, tag = split_tag(ii[idx])
            if is_chunk_start(ii[idx-1], ii[idx]):
                s_pos = idx - 1

            if s_pos != -1:
                word += tt[idx-1]

            if is_chunk_end(ii[idx], ii[idx+1]):
                result.append({'span':(s_pos, idx), 'type':tag, 'disease':word})
                s_pos = -1
                word = ''

        results.append(result)

    return results

def convert_dict_to_xml(sents, dict_list):
    results = []
    for sent, dd in zip(sents, dict_list):
        result = ''
        idx = 0
        for d in dd:
            s_pos, e_pos = d['span']
            while idx < s_pos:
                result += sent[idx]
                idx += 1

            if 'norm' in d:
                result += '<' + d['type'] + ' value="' + d['norm'] + '">'
            else:
                result += '<' + d['type'] + '>'

            result += d['disease']
            result += '</' + d['type'] + '>'

            idx = e_pos

        while idx < len(sent):
            result += sent[idx]
            idx += 1

        results.append(result)

    return results


def convert_iob_to_xml(tokens, iobs):
    results = []
    for tt, ii in zip(tokens, iobs):
        assert len(tt) == len(ii), ''

        result = ''

        ii = ['O'] + ii + ['O']
        for idx in range(1, len(ii)-1):
            prefix, tag = split_tag(ii[idx])
            if is_chunk_start(ii[idx-1], ii[idx]):
                result += '<{}>'.format(tag)

            result += tt[idx-1]

            if is_chunk_end(ii[idx], ii[idx+1]):
                result += '</{}>'.format(tag)

        results.append(result)

    return results

def split_tag(chunk_tag):
    if chunk_tag == 'O':
        return ('O', None)
    else:
        return chunk_tag.split('-', maxsplit=1)


def is_chunk_end(tag, post_tag):
    """
    (current_tag, post_tag)
    (B-C, I-C) -> False
    (B-C, O) -> True
    """

    prefix1, chunk_type1 = split_tag(tag)
    prefix2, chunk_type2 = split_tag(post_tag)

    if prefix1 == 'O':
        return False
    if prefix2 == 'O':
        return prefix1 != 'O'

    return chunk_type1 != chunk_type2


def is_chunk_start(prev_tag, tag):
    """
    (prev_tag, current_tag)
    (B-C, I-C) -> False
    (O, B-C) -> True
    """

    prefix1, chunk_type1 = split_tag(prev_tag)
    prefix2, chunk_type2 = split_tag(tag)

    if prefix2 == 'O':
        return False
    if prefix1 == 'O':
        return prefix2 != 'O'

    return chunk_type1 != chunk_type2

def create_label_vocab_from_file(fn):
    vocab = {}
    with open(fn, 'r') as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue

            if line not in vocab:
                vocab[line] = len(vocab)

    return vocab
