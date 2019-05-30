#!/usr/bin/env python
# -*- coding: utf-8 -*-

from nltk.tokenize import StanfordTokenizer
import argparse
import logging
import numpy as np
import pandas as pd
import re
import sys

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


def segment(texts):
    tk = StanfordTokenizer()
    results = {}
    for text in texts:
        words = tk.tokenize(text)
        segmented = ' '.join(words).lower()
        results[text] = segmented
    return results


def main(args):
    global verbose
    verbose = args.verbose

    cols = ['rel', 'start', 'end', 'text']
    df = pd.read_table(args.path_input, encoding=encoding, names=cols)
    if verbose:
        logger.info('Read {} lines from {}'.format(len(df), args.path_input))
    seg = segment(df['text'].unique())

    df['text'] = df['text'].apply(lambda s: seg.get(s, np.nan))
    n = len(df)
    df = df[pd.notnull(df['text'])]
    assert n == len(df), 'Failed to segment: {} != {}'.format(n, len(df))
    if verbose:
        logger.info('Remained {} rows'.format(len(df)))

    if args.path_output:
        df.to_csv(args.path_output, sep='\t', encoding=encoding,
                  header=False, index=False)
    else:
        df.to_csv(sys.stdout, sep='\t', encoding=encoding,
                  header=False, index=False)
    if verbose:
        logger.info('Write {} lines to {}'.format(
            len(df), args.path_output if args.path_output else 'stdout'))

    return 0


if __name__ == '__main__':
    init_logger('Segment')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
