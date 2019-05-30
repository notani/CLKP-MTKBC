#!/usr/bin/env python
# -*- coding: utf-8 -*-

from conceptnet5.nodes import standardized_concept_uri
from progressbar import ProgressBar
import argparse
import json
import logging
import numpy as np
import pandas as pd
import re

verbose = False
logger = None


def init_logger(name='logger'):
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt)


def filter_by_lang(df, lang):
    prefix = '/c/{}/'.format(lang)
    df = df[df['start'].apply(lambda s: s.startswith(prefix))]
    df = df[df['end'].apply(lambda s: s.startswith(prefix))]
    return df


def extract_surfaceText(s):
    try:
        return json.loads(s)['surfaceText']
    except KeyError:
        return ''


def main(args):
    global verbose
    verbose = args.verbose

    r_term = re.compile(r'\[\[(?P<word>.+?)\]\]')

    CHUNKSIZE = 100000
    cols = ['uri', 'rel', 'start', 'end', 'meta']
    reader = pd.read_table(args.path_input, encoding='utf_8', names=cols,
                           chunksize=CHUNKSIZE)
    df = pd.concat(filter_by_lang(r, args.lang) for r in reader)
    if verbose:
        logger.info('Read {} rows from {}'.format(len(df), args.path_input))

    df['surfaceText'] = df['meta'].apply(extract_surfaceText)
    df.drop(['uri', 'meta'], axis=1, inplace=True)

    indices = df[df['surfaceText'].apply(lambda s: len(s) == 0)].index
    df.drop(indices, axis=0, inplace=True)

    if verbose:
        logger.info('Remained: {} rows'.format(len(df)))
    df.index = np.arange(len(df))

    indices = []
    buff = {'rel': [], 'start': [], 'end': [],
            'surfaceText': [], 'surfaceText-org': []}
    pb = ProgressBar(maxval=len(df))
    for idx, row in df.iterrows():
        surface = row['surfaceText']
        if len(surface) == 0:
            continue
        pb.update(idx+1)
        done = {'start': False, 'end': False}
        _surface = surface
        for m in r_term.finditer(_surface):
            term = standardized_concept_uri(args.lang, m.group('word'))
            for col in ['start', 'end']:
                if term == row[col]:
                    surface = surface.replace('[[{}]]'.format(m.group('word')),
                                              '[[{}]]'.format(col))
                    done[col] = True
                    break
        if done['start'] is False or done['end'] is False:
            indices.append(idx)
            continue
        buff['rel'].append(row['rel'])
        buff['start'].append(row['start'])
        buff['end'].append(row['end'])
        buff['surfaceText'].append(surface)
        buff['surfaceText-org'].append(_surface)

    result = pd.DataFrame.from_dict(buff)
    result.to_csv(args.path_output, sep='\t', encoding='utf_8',
                  index=False)
    if verbose:
        logger.info('Write {} lines to {}'.format(len(result),
                                                  args.path_output))

    return 0


if __name__ == '__main__':
    init_logger('Extract')
    parser = argparse.ArgumentParser()
    parser.add_argument('path_input', help='path to input file')
    parser.add_argument('path_output', help='path to output file')
    parser.add_argument('--lang', default='en',
                        help='target language')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
