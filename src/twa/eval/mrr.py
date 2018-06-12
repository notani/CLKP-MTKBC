#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a Meanrank score."""

import argparse
import pandas as pd
from utils import init_logger


verbose = False
logger = init_logger('MRR')


def read_ranks(filepath):
    df = pd.read_table(filepath, names=['fact_en', 'rank'])
    df['rel'] = df['fact_en'].apply(lambda s: s.split('|||')[0])
    return df


def main(args):
    global verbose
    verbose = args.verbose

    ranks = read_ranks(args.path_input)
    ranks['rank'] = 1.0 / ranks['rank']

    if args.path_output:
        with open(args.path_output, 'w') as f:
            for _, row in ranks.sort_values('fact_en').iterrows():
                f.write('{}\t{}\n'.format(row['fact_en'], row['rank']))

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
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('--by-relation', dest='flag_by_relation',
                        action='store_true', default=False,
                        help='display meanrank by relation')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='verbose output')
    args = parser.parse_args()
    print(main(args))
