#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a Hits@k score."""

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

    ranks['hit'] = 0
    ranks.loc[ranks['rank'] <= args.k, 'hit'] = 1

    if args.path_output:
        with open(args.path_output, 'w') as f:
            for _, row in ranks.sort_values('fact_en').iterrows():
                f.write('{}\t{}\n'.format(row['fact_en'], row['hit']))

    if args.flag_by_relation:
        agg = ranks.groupby('rel')['hit'].agg(['mean', 'size']).sort_index()
        for idx, row in agg.iterrows():
            print('{}\t{}\t{}'.format(idx, row['mean'], int(row['size'])))

    hits = ranks['hit'].mean()
    if verbose:
        logger.info('Hits@{}: {}'.format(args.k, hits))
    return hits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-k', type=int, default=int(10),
                        help='hits@k')
    parser.add_argument('--by-relation', dest='flag_by_relation',
                        action='store_true', default=False,
                        help='display meanrank by relation')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='verbose output')
    args = parser.parse_args()
    print(main(args))
