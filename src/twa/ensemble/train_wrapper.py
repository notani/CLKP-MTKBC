#!/usr/bin/env python
# -*- coding: utf-8 -*-


import argparse
import lzma
import numpy as np
import os

from utils import init_logger
import train

# Directories
dir_scripts = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
dir_root = os.path.dirname(dir_scripts)
dir_data = os.path.join(dir_root, 'data')


verbose = False
logger = init_logger('EnsemberWrapper')


def main(args):
    global verbose
    verbose = args.verbose

    if args.dir_out:
        args.save = True
        if not os.path.isdir(args.dir_out):
            if verbose:
                logger.info('mkdir ' + args.dir_out)
            os.mkdir(args.dir_out)
    else:
        args.save = False
        args.dir_out = 'result-{}'.format(os.getpid())

    results = []
    dir_in = args.dir_split
    dir_out = args.dir_out
    if verbose:
        logger.info(dir_out)
    for _d in os.listdir(dir_in):
        d = os.path.join(dir_in, _d)
        if not os.path.isdir(d):
            continue
        if args.split is not None and int(_d) != args.split:
            if verbose:
                logger.info('Skip {}'.format(d))
            continue
        args.dir_in = d
        args.filename_train = 'train.tsv.xz'
        args.filename_devel = 'devel.tsv.xz'
        if args.flag_comb_train_devel:
            if verbose:
                logger.info('Combine train and devel')
            args.filename_train = 'train-devel.tsv.xz'
            args.filename_devel = 'test.tsv.xz'
            filepath_train = os.path.join(d, args.filename_train)
            if not os.path.isfile(filepath_train):
                if verbose:
                    logger.info('Creating ' + filepath_train)
                with open(filepath_train, 'w') as of:
                    with lzma.open(os.path.join(d, 'train.tsv.xz'), 'rt') as f:
                        for line in f:
                            of.write(line)
                    with lzma.open(os.path.join(d, 'devel.tsv.xz'), 'rt') as f:
                        for line in f:
                            of.write(line)
        if verbose:
            logger.info(args.dir_in)
        args.dir_out = os.path.join(dir_out, _d)
        results.append(train.main(args))

    n_splits = len(results)
    results.append(np.mean(results[:]))

    # Print
    print('split|{}|mean'.format('|'.join(str(v) for v in range(n_splits))))
    print('--|' + '|'.join('--:' for i in range(n_splits + 1)))
    print('ensemble|' + '|'.join('{:.4f}'.format(v) for v in results))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    path_rels = os.path.join(dir_data, 'rel.txt')
    parser.add_argument('dir_split', help='input directory')
    parser.add_argument('-o', '--output', dest='dir_out',
                        help='output directory')
    parser.add_argument('-b', '--batchsize', dest='batch_size',
                        type=int, default=int(64),
                        help='learning minibatch size')
    parser.add_argument('-e', '--epoch', dest='n_epochs',
                        type=int, default=int(30),
                        help='number of epochs to learn')
    parser.add_argument('-m', '--model', dest='model',
                        default='linear', help='model architecture')
    parser.add_argument('--lr', type=float, default=float(0.1),
                        help='learning rate')
    parser.add_argument('--wd', type=float, help='weight decay')
    parser.add_argument('--lam', type=float, help='L2 penalty')
    parser.add_argument('--margin', dest='margin', default=1.0, type=float,
                        help='margin in a loss function')
    parser.add_argument('--degree', dest='poly_degree', default=1, type=int,
                        help='degree of polynomial transformation')
    parser.add_argument('--uniform-weight', dest='flag_unifw',
                        action='store_true', default=False,
                        help='use uniform weight over all rels'
                        'in terms of meanrank on the devel set')
    parser.add_argument('--standardize', dest='flag_standardize',
                        action='store_true', help='standardize vectors')
    parser.add_argument('--comb', dest='flag_comb_train_devel',
                        action='store_true', default=False,
                        help='combine train.tsv and devel.tsv')
    parser.add_argument('--split', type=int,
                        help='split number to be used')
    parser.add_argument('--path-rels', dest='path_rels',
                        default=path_rels)
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
