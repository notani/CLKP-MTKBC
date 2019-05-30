#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import cuda, Chain
from itertools import chain
from os import path
import argparse
import chainer
import numpy as np

from dataset import COL_BASIC_FEATURES
from dataset import FactIterator, Vocabulary
from dataset import read_dataset, get_idx2vec, get_values, replace_by_dic
from models import LinearEnsembler, MLPEnsembler
from utils import init_logger
from utils import standardize_vectors

verbose = False
logger = init_logger('Rank')
encoding = 'utf_8'

# Directories
dir_scripts = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
dir_root = path.dirname(dir_scripts)
dir_data = path.join(dir_root, 'data')


class Classifier(Chain):
    """Calculate loss values."""

    def __init__(self, predictor, label2fact, en2ja, idx2vec):
        """Construct a Classifier.
        Args:
            predictor: model
        """
        self.idx2vec = idx2vec
        self.en2ja = {k: np.array(list(v), dtype=np.int32)
                      for k, v in en2ja.items()}
        self.ja_cands = {k: np.array(list(v.intersection(label2fact[0])),
                                     np.int32)
                         for k, v in en2ja.items()}
        self.n_ja_cands_max = max(len(v) for v in self.ja_cands.values())
        self.ja_golds = {k: v.intersection(label2fact[1])
                         for k, v in en2ja.items()}
        super(Classifier, self).__init__(predictor=predictor)

    def __call__(self, rels, en_indices):
        def evaluate():
            rels_ = xp.concatenate(buff['rel'])
            vecs_ = xp.concatenate(buff['vec'])
            scores_ = xp.split(self.predictor(rels_, vecs_).data,
                               splitter[:-1])
            if xp is not np:
                for i in range(len(buff['idx'])):
                    buff['idx'][i] = cuda.to_cpu(buff['idx'][i])
                    scores_[i] = cuda.to_cpu(scores_[i])
            memo = []
            for i, (indices, scores) in enumerate(zip(buff['idx'], scores_)):
                itr = sorted(zip(indices, scores),
                             key=lambda t: t[1], reverse=True)
                for rank, (ja, _) in enumerate(itr, start=1):
                    if ja in buff['gold'][i]:
                        memo.append(rank)
                        break
            assert len(memo) == len(scores_), '{} != {}'.format(
                len(memo), len(scores_))
            return memo

        xp = cuda.get_array_module(rels)
        BUFF_LIMIT = 1e+4

        buff = {'gold': [], 'idx': [], 'vec': [], 'rel': []}
        splitter = []
        n_buff = 0
        ranks = []
        for rel, en in zip(rels, en_indices):
            buff['gold'].append(self.ja_golds[en])
            indices = self.en2ja[en]
            buff['idx'].append(indices)
            buff['vec'].append(self.idx2vec[indices])
            rels_ = xp.empty(self.idx2vec[indices].shape[0],
                             dtype=xp.int32)
            rels_[:] = rel
            buff['rel'].append(rels_)
            n_buff += len(self.en2ja[en])
            splitter.append(n_buff)

            if n_buff < BUFF_LIMIT:
                continue

            ranks += evaluate()
            buff = {'gold': [], 'idx': [], 'vec': [], 'rel': []}
            n_buff = 0
            splitter = []

        if buff['gold']:
            ranks += evaluate()
        return ranks


def main(args):
    global verbose, encoding
    verbose = args.verbose
    encoding = args.encoding
    assert args.poly_degree >= 1, '--degree must be positive integer'
    poly_degree = args.poly_degree

    gpu = args.gpu
    if gpu >= 0:
        cuda.check_cuda_available()
        if verbose:
            logger.info('Use GPU {}'.format(gpu))
        cuda.get_device_from_id(gpu).use()

    df = read_dataset(args.path_input, args.flag_has_header, args.threshold)

    agg = df.groupby('fact_en')['twa'].mean()
    invalid_facts = set(agg[(agg == 1.0)|(agg == 0.0)].index)
    if verbose:
        logger.info('Invalid facts: {}'.format(len(invalid_facts)))
    df = df[~df['fact_en'].isin(invalid_facts)]
    if verbose:
        logger.info('Remained {} lines'.format(len(df)))

    # Load vocabulary
    if verbose:
        logger.info('Load vocabulary')
    rel2id = Vocabulary()
    rel2id.read_from_file(args.path_rels)
    fact2id = Vocabulary()
    fact2id.read_from_list(np.unique(get_values(df, 'fact')))
    ja2id = Vocabulary()
    ja2id.read_from_list(np.unique(get_values(df, 'fact_ja')))
    en2id = Vocabulary()
    en2id.read_from_list(np.unique(get_values(df, 'fact_en')))

    df.loc[:, 'fact'] = replace_by_dic(df['fact'], fact2id).astype(np.int32)
    df.loc[:, 'fact_ja'] = replace_by_dic(df['fact_ja'], ja2id).astype(np.int32)
    df.loc[:, 'fact_en'] = replace_by_dic(df['fact_en'], en2id).astype(np.int32)
    df.loc[:, 'rel'] = replace_by_dic(df['rel'], rel2id).astype(np.int32)

    label2fact = {i: set(df[df['twa'] == i]['fact'].unique())
                  for i in [0, 1]}
    en2ja = {en: set(df[df['fact_en'] == en]['fact'].unique())
             for en in sorted(df['fact_en'].unique())}
    idx2vec = get_idx2vec(df, poly_degree=poly_degree)
    if gpu >= 0:
        idx2vec = cuda.to_gpu(idx2vec)

    ss = df[df['twa'] == 1].drop_duplicates('fact_en')
    itr = FactIterator(ss, len(ss), ja2id, en2id, train=False, repeat=False,
                       poly_degree=poly_degree)

    # Define a model
    model_type = args.model.lower()
    dim_in = len(COL_BASIC_FEATURES)
    rel_size = len(rel2id)
    if model_type.startswith('linear'):
        ensembler = LinearEnsembler(dim_in, rel_size, use_gpu=(gpu >= 0),
                                    poly_degree=poly_degree,
                                    flag_unifw=args.flag_unifw,
                                    verbose=verbose)
    elif model_type.startswith('mlp'):
        options = args.model.split(':')
        params = {}
        if len(options) > 1:
            params['dim_hid'] = int(options[1])
        if len(options) > 2:
            params['activation'] = options[2]
        ensembler = MLPEnsembler(
            dim_in, rel_size, use_gpu=(gpu >= 0),
            poly_degree=poly_degree, flag_unifw=args.flag_unifw,
            verbose=verbose, **params)
    else:
        raise ValueError('Invalid --model: {}'.format(model_type))

    ensembler.add_persistent('_mu', None)
    ensembler.add_persistent('_sigma', None)
    # load a trained model
    chainer.serializers.load_npz(args.path_model, ensembler)
    if ensembler._mu is not None:
        logger.info('standardize vectors: True')
        itr.standardize_vectors(mu=ensembler._mu, sigma=ensembler._sigma)
        idx2vec = standardize_vectors(idx2vec, ensembler._mu, ensembler._sigma)
    else:
        logger.info('standardize vectors: False')

    model = Classifier(ensembler, label2fact, en2ja, idx2vec)

    # calculate probabilities for testing set
    buff = []
    for i, (rels, _, en_indices) in enumerate(itr, start=1):
        if i % 500 == 0:
            logger.info('Evaluating: {}'.format(i))
        buff.append((model(rels, en_indices), en_indices))
    ranks = list(chain.from_iterable(t[0] for t in buff))
    facts_en = [en2id.id2word[idx]
                for idx in chain.from_iterable(t[1] for t in buff)]

    if verbose:
        logger.info('Output results to ' + args.path_output)
    with open(args.path_output, 'w') as f:
        for fact, rank in sorted(zip(facts_en, ranks), key=lambda t: t[0]):
            f.write('{}\t{}\n'.format(fact, rank))


if __name__ == '__main__':
    init_logger()
    parser = argparse.ArgumentParser()
    path_rels = path.join(dir_data, 'rel.txt')
    parser.add_argument('-i', '--input', dest='path_input',
                        required=True, help='path to input file')
    parser.add_argument('-m', '--model', dest='path_model',
                        required=True, help='path to model file')
    parser.add_argument('--model-type', dest='model',
                        default='linear', help='model architecture')
    parser.add_argument('-e', '--encoding', default='utf_8',
                        help='I/O Encoding')
    parser.add_argument('-o', '--output', dest='path_output',
                        required=True,
                        help='path to output file')
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
    parser.add_argument('--has-header', dest='flag_has_header',
                        action='store_true', default=False,
                        help='has header')
    parser.add_argument('-t', '--threshold', type=int, default=1)
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
