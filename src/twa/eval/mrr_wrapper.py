#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Calculate MRR for all method on all split.

Example:
    nice -n 19 python scripts/twa/eval/mrr_wrapper.py /baobab/otani/cckbc/cn/20170814v1/ens/kbc-170816v2/split -v --by-relation
"""

import argparse
import numpy as np
import os

import mrr
from utils import init_logger


verbose = False
logger = init_logger('EvalMrr')


def main(args):
    global verbose
    verbose = args.verbose

    assert args.label in ['devel', 'test']

    methods = ['pmi', 'kbc', 'trans', 'kbc-trans']
    if args.method:
        methods = [args.method]
    results = {m: [] for m in methods}
    for d in os.listdir(args.dir_split):
        d = os.path.join(args.dir_split, d)
        if not os.path.isdir(d):
            continue
        for method in results.keys():
            filename = 'pred.{}.{}.tsv.xz'.format(method, args.label)
            args.path_input = os.path.join(d, filename)
            if verbose:
                logger.info(args.path_input)
            args.method = method
            results[method].append(mrr.main(args))

    n_splits = len(results[methods[0]])
    for method in results.keys():
        results[method].append(np.mean(results[method]))

    # Print
    print('split|{}|mean'.format('|'.join(str(v) for v in range(n_splits))))
    print('--|' + '|'.join('--:' for i in range(n_splits + 1)))
    for method in methods:
        print(method + '|' + '|'.join('{:.3f}'.format(v)
                                      for v in results[method]))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir_split', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-m', '--method',
                        help='specific type of scoring method')
    parser.add_argument('--label', default='devel',
                        help='label (devel|test)')
    parser.add_argument('--by-relation', dest='flag_by_relation',
                        action='store_true', default=False,
                        help='display score by relation')
    parser.add_argument('--has-header', dest='flag_has_header',
                        action='store_true', default=False,
                        help='has header')
    parser.add_argument('-t', '--threshold', type=int, default=3)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
