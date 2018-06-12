#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Output ranks."""

from sklearn.preprocessing import MinMaxScaler
import argparse
import numpy as np
import pandas as pd

from utils import init_logger


verbose = False
logger = init_logger('Rank')


def read_dataset(filepath, has_header=False, threshold=None):
    if has_header:
        df = pd.read_table(filepath, engine='python')
    else:
        cols = ['uri', 'rel', 'start', 'end', 'start_en', 'end_en',
                'meta', 'freq_start', 'freq_end', 'co-freq',
                'pmi', 'trans', 'twa', 'kbc']
        df = pd.read_table(filepath, names=cols, header=None, engine='python')

    if 'twa' not in df.columns:
        assert threshold is not None
        assert 'label' in df.columns
        df.loc[:, 'twa'] = 0
        df.loc[df['label'] <= threshold, 'twa'] = 1
    if verbose:
        logger.info('Read {} lines from {}'.format(len(df), filepath))
    _ = '|||'
    df.loc[:, 'fact_ja'] = df['rel'] + _ + df['start'] + _ + df['end']
    df.loc[:, 'fact_en'] = df['rel'] + _ + df['start_en'] + _ + df['end_en']
    cols = ['uri', 'rel', 'start', 'end', 'start_en', 'end_en', 'meta']
    if 'label' in df.columns:
        cols.append('label')
    return df.drop(cols, axis=1)


def drop_invalid_twa(df):
    agg = df.groupby('fact_en')['twa'].mean()
    invalid_facts = set(agg[(agg == 1.0) | (agg == 0.0)].index)
    if verbose:
        logger.info('Invalid facts: {}'.format(len(invalid_facts)))
    df = df[~df['fact_en'].isin(invalid_facts)]
    if verbose:
        logger.info('Remained {} lines'.format(len(df)))
    return df


def get_ranks(df, method):
    ranks = []
    facts_en = []
    if method == 'pmi':  # PPMI
        indices_invalid = df['pmi'] == 0  # invalid PMI
        # invalid PMI with positive label
        indices_invalid_1 = indices_invalid & (df['twa'] == 1)
        df.loc[df['pmi'] < 0, 'pmi'] = 0.0
        df.loc[df[indices_invalid].index, 'pmi'] = -1.0
        df.loc[df[indices_invalid_1].index, 'pmi'] = -2.0
    if method == 'kbc-trans':  # KBC+Trans
        df['kbc'] = MinMaxScaler().fit_transform(
            df['kbc'].values.reshape((-1, 1))).flatten()
        df['trans'] = MinMaxScaler().fit_transform(
            df['trans'].values.reshape((-1, 1))).flatten()
        df['kbc-trans'] = df['kbc'] + df['trans']
    # Filtering
    df = drop_invalid_twa(df).copy()
    n_facts = len(df.index.unique())
    m = n_facts // 10
    df.sort_values([method, 'twa'], ascending=[False, False],
                   inplace=True)
    df.set_index('fact_en', inplace=True)
    for i, fact_en in enumerate(sorted(df.index.unique()), start=1):
        if verbose and (i % m == 0):
            logger.info('{}/{} [{}]'.format(i, n_facts, int(i / m)))
        ss = df.loc[fact_en]
        ranks.append(np.where(ss['twa'].values == 1)[0][0] + 1)
        facts_en.append(fact_en)
    return ranks, facts_en


def output_ranks(filepath, ranks, facts_en):
    if verbose:
        logger.info('Output results to ' + filepath)
    print(filepath)
    with open(filepath, 'w') as f:
        for fact, rank in sorted(zip(facts_en, ranks),
                                 key=lambda t: t[0]):
            f.write('{}\t{}\n'.format(fact, rank))


def main(args):
    global verbose
    verbose = args.verbose

    method = args.method.lower()
    assert method in ['pmi', 'kbc', 'trans', 'transe', 'kbc-trans', 'other'], \
        'Invalid scoring method: ' + method
    if verbose:
        logger.info('Method: ' + method)

    # Read
    df = read_dataset(args.path_input, args.flag_has_header, args.threshold)

    if method == 'other':
        if args.path_score is None:
            logger.error('--score is required')
            return 1
        df[method] = np.loadtxt(args.path_score)

    ranks, facts_en = get_ranks(df, method)

    assert len(ranks) == len(facts_en)

    # Write
    output_ranks(args.path_output, ranks, facts_en)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('method', help='scoring method (pmi|kbc|trans|other)')
    parser.add_argument('--score', dest='path_score',
                        help='path to score file')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True, help='path to output file')
    parser.add_argument('--has-header', dest='flag_has_header',
                        action='store_true', default=False,
                        help='has header')
    parser.add_argument('-t', '--threshold', type=int, default=1)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
