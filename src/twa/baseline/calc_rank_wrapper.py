#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Output ranks for all method on all split.

Example:
    python scripts/twa/postprocessing/calc_rank_wrapper.py /baobab/otani/cckbc/cn/20170612v1/pred-170606-iter_2748/split -v
"""

import argparse
import os

import calc_rank
from utils import init_logger


verbose = False
logger = init_logger('Rank')


def main(args):
    global verbose
    verbose = args.verbose

    assert args.filename in ['devel.tsv', 'test.tsv',
                             'devel.tsv.xz', 'test.tsv.xz']

    if args.dir_output is None:
        args.dir_output = args.dir_split

    for d in os.listdir(args.dir_split):
        d = os.path.join(args.dir_output, d)
        if not os.path.isdir(d):
            continue

        filepath = os.path.join(d, args.filename)
        if verbose:
            logger.info(filepath)

        args.path_input = filepath
        for method in ['pmi', 'kbc', 'trans', 'kbc-trans']:
            args.path_output = os.path.join(
                    d, 'pred.{}.{}.tsv'.format(method, args.filename.split('.')[0]))
            args.method = method
            calc_rank.main(args)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_split', help='path to input directory')
    parser.add_argument('--filename', default='devel.tsv',
                        help='filename (devel.tsv[.xz]|test.tsv[.xz])')
    parser.add_argument('-o', '--output', dest='dir_output',
                        help='path to output directory [=dir_split]')
    parser.add_argument('--has-header', dest='flag_has_header',
                        action='store_true', default=False,
                        help='has header')
    parser.add_argument('-t', '--threshold', type=int, default=3)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
