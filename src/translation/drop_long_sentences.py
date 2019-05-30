#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import sys

verbose = False
logger = None
encoding = 'utf_8'


def init_logger(name='logger'):
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def main(args):
    global verbose, encoding
    verbose = args.verbose
    encoding = args.encoding

    count = [0, 0]
    for i, line in enumerate(sys.stdin, start=1):
        if verbose and i % 100000 == 0:
            logger.info(i)
        sent1, sent2 = line.split('\t')
        N = len(sent1.strip().split())
        if N > args.limit:
            count[0] += 1
            continue
        N = len(sent2.strip().split())
        if N > args.limit:
            count[0] += 1
            continue
        sys.stdout.write(line)
        count[1] += 1

    if verbose:
        logger.info('Dropped {} sentences. Remained {} sentences'.format(
            count[0], count[1]))
    return 0


if __name__ == '__main__':
    init_logger('Drop')
    parser = argparse.ArgumentParser()
    parser.add_argument('limit', type=int,
                        help='upper bound of sentence length')
    parser.add_argument('-e', '--encoding', default='utf_8',
                        help='I/O Encoding')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
