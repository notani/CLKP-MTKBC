#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a Meanrank score."""

import argparse
import pandas as pd
from utils import init_logger


verbose = False
logger = init_logger('Meanrank')


def read_ranks(filepath):
    df = pd.read_table(filepath, names=['fact_en', 'rank'])
    df['rel'] = df['fact_en'].apply(lambda s: s.split('|||')[0])
    return df


def main(args):
    global verbose
    verbose = args.verbose

    ranks = read_ranks(args.path_input)

    if args.flag_by_relation:
        agg = ranks.groupby('rel')['rank'].agg(['mean', 'size']).sort_index()
        for idx, row in agg.iterrows():
            print('{}\t{}\t{}'.format(idx, row['mean'], int(row['size'])))

    meanrank = ranks['rank'].mean()
    if verbose:
        logger.info('Meanrank: {}'.format(meanrank))
    return meanrank


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('--by-relation', dest='flag_by_relation',
                        action='store_true', default=False,
                        help='display meanrank by relation')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='verbose output')
    args = parser.parse_args()
    print(main(args))
