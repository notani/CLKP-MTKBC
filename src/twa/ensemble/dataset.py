#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script of dataset manager"""


import chainer
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

from utils import standardize_vectors

COL_BASIC_FEATURES = ['kbc', 'trans']


def read_dataset(filepath, has_header=False, threshold=None):
    if has_header:
        data = pd.read_table(filepath, engine='python')
    else:
        cols = ['uri', 'rel', 'start', 'end', 'start_en', 'end_en',
                'meta', 'freq_start', 'freq_end', 'co-freq',
                'pmi', 'trans', 'twa', 'kbc']
        data = pd.read_table(filepath, names=cols, header=None, engine='python')

    cols = ['fact', 'fact_ja', 'fact_en', 'rel', 'pmi', 'kbc', 'trans', 'twa']
    if threshold and 'twa' not in data.columns:
        assert 'label' in data.columns
        data.loc[:, 'twa'] = 0
        data.loc[data['label'] <= threshold, 'twa'] = 1
    elif 'label' in data.columns:
        cols = ['fact', 'fact_ja', 'fact_en', 'rel', 'pmi', 'kbc', 'trans', 'label']
    data.loc[:, 'fact_ja'] = data['rel'] + '|||' + data['start'] + '|||' + data['end']
    data.loc[:, 'fact_en'] = data['rel'] + '|||' + data['start_en'] + '|||' + data['end_en']
    data.loc[:, 'rel'] = data['rel'].apply(lambda s: s.split('/')[-1].lower())
    data.loc[:, 'fact'] = data['fact_ja'] + '@@@' + data['fact_en']
    return data[cols]


def get_idx2vec(dfs, poly_degree=None):
    """Return mapping of fact index to feature vector"""
    if not type(dfs) is list:
        dfs = [dfs]
    df = pd.concat(dfs).drop_duplicates('fact').sort_values('fact')
    vecs = df[COL_BASIC_FEATURES].values
    if type(poly_degree) is int:
        poly = PolynomialFeatures(poly_degree, include_bias=False)
        vecs = poly.fit_transform(vecs[:])
    return np.r_[np.ones((1, vecs.shape[1])),  # UNK
                 vecs].astype(np.float32)


def get_values(dfs, col):
    """Return a list of values of col in dfs"""
    if not type(dfs) is list:
        dfs = [dfs]
    values = []
    for df in dfs:
        values += df[col].tolist()
    return values


def replace_by_dic(ser, dic):
    def replace(s):
        return dic[s]
    return ser.apply(replace)


class Vocabulary(object):
    """Manage vocabulary."""

    def __init__(self):
        """Initialize vocabulary."""
        self.ID_UNK = 0
        self.id2word = ['<UNK>']
        self.word2id = {'<UNK>': 0}

    def read_from_list(self, l):
        self.id2word += sorted(l)
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

    def read_from_file(self, filepath):
        """Read words from a file."""
        with open(filepath) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                word = line.strip()
                if word in self.word2id:
                    continue
                self.word2id[word] = len(self.word2id)
                self.id2word.append(word)

    def __getitem__(self, word):
        try:
            return self.word2id[word]
        except KeyError:
            return self.ID_UNK

    def get_ids(self, word, delimiter=None):
        """Return ID as a list"""
        if type(delimiter) is str:
            return [self[w] for w in word.split(delimiter)]
        return [self[word]]

    def __len__(self):
        return len(self.id2word) - 1

    def ids(self):
        return self.word2id.values()

    def print_vocab(self):
        for i, word in enumerate(self.id2word):
            print('{}\t{}'.format(i, word))


class FactIterator(chainer.dataset.Iterator):
    """Iterator of facts"""

    def __init__(self, dataset, batch_size, ja2id, en2id,
                 train=True, evaluate=False, repeat=True, poly_degree=None):
        "Initialize an iterator"
        if not evaluate:
            dataset = dataset[dataset['twa'] == 1]

        self.ja2id = ja2id
        self.en2id = en2id
        self.size = len(dataset)
        self.batch_size = batch_size
        if self.batch_size <= 0:
            self.batch_size = self.size
        self.epoch = 0
        self.iteration = 0
        self.train = train
        self.repeat = repeat
        self.poly_degree = poly_degree
        self.is_new_epoch = False

        self.load_dataset(dataset)
        if self.train:
            self.shuffle()

    def load_dataset(self, dataset):
        """Load data as a numpy array."""
        self.rels = dataset['rel'].values.astype(np.int32)
        self.vecs = dataset[COL_BASIC_FEATURES].values.astype(np.float32)
        if type(self.poly_degree) is int:
            poly = PolynomialFeatures(self.poly_degree, include_bias=False)
            self.vecs = poly.fit_transform(self.vecs[:])
        self.en_indices = dataset['fact_en'].values

    def __next__(self):
        """Return next batch"""
        if not self.repeat and self.iteration * self.batch_size >= self.size:
            raise StopIteration

        rels, vecs, en_indices = self.get_facts()
        self.iteration += 1

        epoch = self.iteration * self.batch_size // self.size
        self.is_new_epoch = self.epoch < epoch
        if self.is_new_epoch:
            self.epoch = epoch
            if self.train:
                self.shuffle()

        return rels, vecs, en_indices

    def get_facts(self):
        """Get facts that construct batch data.

        Returns:
        - rels: array of int
        - vecs: 2-dimensional vector
        - en_indices: array of int
        """
        # calculate an offset
        offset = self.iteration
        if self.repeat:
            max_offset = self.size // self.batch_size
            if self.size % self.batch_size > 0:
                max_offset += 1
            offset %= max_offset
        # get index which is use for selecting sample
        batch_size = min(self.batch_size, self.size - self.batch_size * offset)
        indices = [i + self.batch_size * offset for i in range(batch_size)]
        return self.rels[indices], self.vecs[indices], self.en_indices[indices]

    def shuffle(self):
        """Shuffle dataset"""
        perm = np.random.permutation(self.size)
        self.rels[...] = self.rels[perm]
        self.vecs[...] = self.vecs[perm]
        self.en_indices[...] = self.en_indices[perm]

    def standardize_vectors(self, mu=None, sigma=None):
        """Standardize vectors."""
        if mu is None:
            mu = self.vecs.mean(axis=0)
            sigma = self.vecs.std(axis=0)
        self.vecs = standardize_vectors(self.vecs, mu, sigma)
        return mu, sigma

    @property
    def epoch_detail(self):
        """Report progress"""
        return self.iteration * self.batch_size / self.size

    def serialize(self, serializer):
        """Update progress information"""
        self.iteration = serializer('iteration', self.iteration)
        self.epoch = serializer('epoch', self.epoch)
