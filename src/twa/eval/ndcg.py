#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluate a nDCG@k score."""

from sklearn.preprocessing import MinMaxScaler
import argparse
import pandas as pd

from rank_metrics import ndcg_at_k
from utils import init_logger


verbose = False
logger = init_logger('Meanrank')


def read_scores(filepath, col=None, reverse=False):
    if col is None:
        col = 'score'
    df = pd.read_table(filepath)
    df['fact_en'] = df['start_en'] + '|||' + df['rel'] + '|||' + df['end_en']

    if col == 'pmi':  # PPMI
        df.loc[df['pmi'] < 0, 'pmi'] = 0.0

    if col == 'kbc-trans':  # KBC+Trans
        df['kbc'] = MinMaxScaler().fit_transform(
            df['kbc'].values.reshape((-1, 1))).flatten()
        df['trans'] = MinMaxScaler().fit_transform(
            df['trans'].values.reshape((-1, 1))).flatten()
        df['kbc-trans'] = df['kbc'] + df['trans']

    if reverse:
        label_max = df['label'].max()
        df.loc[:, 'label'] = label_max - df.loc[:, 'label']
    else:
        label_min = df['label'].min()
        df.loc[:, 'label'] = df['label'] - label_min

    df.sort_values([col, 'label'], ascending=[False, True], inplace=True)
    df.set_index('fact_en', inplace=True)

    return df


def main(args):
    global verbose
    verbose = args.verbose

    scores = read_scores(args.path_input, col=args.col,
                         reverse=args.flag_reverse)

    rel2facts = {}
    ndcgs = {}
    for fact_en in scores.index.unique():
        ss = scores.loc[fact_en]
        try:
            rel = ss['rel'].values[0]
        except:
            logger.warning('Only one Japanese fact')
            continue
        try:
            rel2facts[rel].append(fact_en)
        except KeyError:
            rel2facts[rel] = [fact_en]
        ndcgs[fact_en] = ndcg_at_k(ss['label'].values,
                                   args.k, method=0)

    if args.path_output:
        with open(args.path_output, 'w') as f:
            for k, v in sorted(ndcgs.items(), key=lambda t: t[0]):
                f.write('{}\t{}\n'.format(k, v))

    if args.flag_by_relation:
        for rel, facts in sorted(rel2facts.items(),
                                 key=lambda t: t[0]):
            l = [ndcgs[fact_en] for fact_en in facts]
            print('{}\t{}\t{}'.format(rel, sum(l) / len(l), len(l)))

    ndcg = sum(ndcgs.values()) / len(ndcgs)
    if verbose:
        logger.info('nDCG@{}: {}'.format(args.k, ndcg))
    return ndcg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('-o', '--output', dest='path_output',
                        help='path to output file')
    parser.add_argument('-k', type=int, default=int(10),
                        help='nDCG@k')
    parser.add_argument('--col', help='score columns')
    parser.add_argument('--by-relation', dest='flag_by_relation',
                        action='store_true', default=False,
                        help='display meanrank by relation')
    parser.add_argument('--reverse', dest='flag_reverse',
                        action='store_true', default=False,
                        help='reverse labels')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='verbose output')
    args = parser.parse_args()
    print(main(args))
