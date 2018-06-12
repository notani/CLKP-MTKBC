#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import numpy as np
import os

from utils import init_logger
import calc_rank

verbose = False
logger = init_logger('RankWrapper')
encoding = 'utf_8'

# Directories
dir_cur = os.path.abspath(__file__)
dir_scripts = os.path.dirname(os.path.dirname(os.path.dirname(dir_cur)))
dir_root = os.path.dirname(dir_scripts)
dir_data = os.path.join(dir_root, 'data')


def main(args):
    global verbose, encoding
    verbose = args.verbose
    encoding = args.encoding

    assert args.label in ['devel', 'test']

    args.flag_has_header = False
    args.threshold = 1

    dir_model = args.dir_model
    dir_in = args.dir_in
    for _d in os.listdir(dir_in):
        d = os.path.join(dir_in, _d)
        if not os.path.isdir(d):
            continue
        args.path_input = os.path.join(d, '{}.tsv.xz'.format(args.label))
        args.path_model = os.path.join(os.path.join(dir_model, _d),
                                       'ensemble.model')
        args.path_output = os.path.join(os.path.join(dir_model, _d),
                                        'pred.ens.{}.tsv'.format(args.label))
        calc_rank.main(args)

    return 0


if __name__ == '__main__':
    init_logger()

    parser = argparse.ArgumentParser()
    path_rels = os.path.join(dir_data, 'rel.txt')
    parser.add_argument('-i', '--input', dest='dir_in',
                        required=True, help='path to input directory')
    parser.add_argument('-m', '--model', dest='dir_model',
                        required=True, help='path to model directory')
    parser.add_argument('--model-type', dest='model',
                        default='linear', help='model architecture')
    parser.add_argument('--label', default='devel',
                        help='label (devel|test)')
    parser.add_argument('-e', '--encoding', default='utf_8',
                        help='I/O Encoding')
    parser.add_argument('--path-rels', dest='path_rels',
                        default=path_rels)
    parser.add_argument('--uniform-weight', dest='flag_unifw',
                        action='store_true', default=False,
                        help='use uniform weight over all rels'
                        'in terms of meanrank on the devel set')
    parser.add_argument('--degree', dest='poly_degree', default=1, type=int,
                        help='degree of polynomial transformation')
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
