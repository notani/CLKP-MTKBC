#!/usr/bin/env python
# -*- coding: utf-8 -*-

from mojimoji import han_to_zen
from nltk.tokenize import StanfordTokenizer
from pyknp import Jumanpp
import pdb
import argparse
import logging
import numpy as np
import pandas as pd
import re

encoding = 'utf_8'
verbose = False
logger = None

r_num = re.compile(r'[0-9]+')


def init_logger(name='logger'):
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def replace_number(s):
    return r_num.sub('<NUM>', s)


def segment_ja(texts, flag_keep_number=False):
    jumanpp = Jumanpp()
    results = {}
    for text in texts:
        try:
            parsed = jumanpp.analysis(han_to_zen(text))
            if flag_keep_number:
                segmented = ' '.join(m.midasi for m in parsed.mrph_list())
            else:
                segmented = ' '.join('<数詞>' if m.bunrui == '数詞' else m.midasi
                                     for m in parsed.mrph_list())
            results[text] = segmented
        except Exception:
            pdb.set_trace()
            logger.warning('Cannot parse {}'.format(text))
            continue
    return results


def segment_en(texts, flag_keep_number=False):
    tk = StanfordTokenizer()
    results = {}
    for text in texts:
        if flag_keep_number:
            words = tk.tokenize(text)
        else:
            words = map(replace_number, tk.tokenize(text))
        segmented = ' '.join(words).lower()
        results[text] = segmented
    return results


def main(args):
    global verbose
    verbose = args.verbose

    df = pd.read_table(args.path_input, encoding=encoding)
    if verbose:
        logger.info('Read {} lines from {}'.format(len(df), args.path_input))
    seg_ja = segment_ja(df['text_ja'].unique(),
                        flag_keep_number=args.flag_keep_number)
    seg_en = segment_en(df['text_en'].unique(),
                        flag_keep_number=args.flag_keep_number)

    df['text_ja_seg'] = df['text_ja'].apply(lambda s: seg_ja.get(s, np.nan))
    df = df[pd.notnull(df['text_ja_seg'])]
    df['text_en_seg'] = df['text_en'].apply(lambda s: seg_en.get(s, np.nan))
    df = df[pd.notnull(df['text_ja_seg'])]
    if verbose:
        logger.info('Remained {} rows'.format(len(df)))

    df.to_csv(args.path_output, sep='\t', encoding=encoding,
              header=False, index=False)
    if verbose:
        logger.info('Write {} lines to {}'.format(len(df), args.path_output))

    return 0


if __name__ == '__main__':
    init_logger('Segment')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('path_output', help='path to output file')
    parser.add_argument('--keep-number', dest='flag_keep_number',
                        action='store_true', default=False,
                        help='keep numbers')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
