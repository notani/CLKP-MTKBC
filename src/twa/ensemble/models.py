#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain, Variable
import chainer.functions as F
import chainer.links as L
from itertools import combinations_with_replacement
from utils import init_logger


class LinearEnsembler(Chain):
    def __init__(self, dim_in, rel_vocab_size, use_gpu=False,
                 poly_degree=None, flag_unifw=False,
                 verbose=False):
        "Initialize a model."

        self.verbose = verbose
        self.logger = init_logger(self.__class__.__name__)

        self.use_gpu = use_gpu
        self.rel_vocab_size = rel_vocab_size
        self.n_features = dim_in
        if type(poly_degree) is int:
            for i in range(1, poly_degree):
                # TODO: refactoring; the code below is not efficient
                combs = combinations_with_replacement(range(dim_in), i+1)
                self.n_features += len(list(combs))
        self.poly_degree = poly_degree
        self.flag_unifw = flag_unifw

        super(LinearEnsembler, self).__init__(
            w=L.EmbedID(rel_vocab_size + 1, self.n_features),
            b=L.EmbedID(rel_vocab_size + 1, 1)
        )

        if self.verbose:
            self.logger.info(
                'LinearEnsembler n_features={n_fea}'.format(
                    n_fea=self.n_features))

    def __call__(self, rels, vecs):
        "Return a score of fact; closer to zero is better"
        batch_size = rels.shape[0]
        shape = (batch_size, -1)

        xp = self.xp
        if self.flag_unifw:
            rels = xp.zeros(rels.shape)

        bias = self.b(rels)
        weight = self.w(rels)
        X = Variable(vecs)

        return F.batch_matmul(X, weight, transa=True).reshape(shape) \
            + bias.reshape(shape)

    @property
    def theta(self):
        """Return all parameters"""
        xp = self.xp
        indices = xp.arange(1, self.rel_vocab_size+1, dtype=xp.int32)
        return F.flatten(self.w(indices))


class MLPEnsembler(Chain):
    """MLP ensembler."""
    def __init__(self, dim_in, rel_vocab_size, dim_hid=16, activation='sigmoid',
                 use_gpu=False, poly_degree=None, flag_unifw=False,
                 verbose=False):
        "Initialize a model."

        self.verbose = verbose
        self.logger = init_logger(self.__class__.__name__)

        self.use_gpu = use_gpu
        self.rel_vocab_size = rel_vocab_size
        self.n_features = dim_in
        if type(poly_degree) is int:
            for i in range(1, poly_degree):
                # TODO: refactoring; the code below is not efficient
                combs = combinations_with_replacement(range(dim_in), i+1)
                self.n_features += len(list(combs))
        self.dim_hid = dim_hid
        if activation == 'sigmoid':
            self.g = F.sigmoid
        elif activation == 'tanh':
            self.g = F.tanh
        elif activation == 'relu':
            self.g = F.relu
        else:
            raise ValueError("activation='{}' is invalid".format(
                activation))

        self.poly_degree = poly_degree
        self.flag_unifw = flag_unifw
        super(MLPEnsembler, self).__init__(
            f=L.Linear(self.n_features, self.dim_hid),
            w=L.EmbedID(rel_vocab_size + 1, self.dim_hid),
            b=L.EmbedID(rel_vocab_size + 1, 1)
        )

        if self.verbose:
            self.logger.info('MLPEnsembler '
                             'n_features:={n_fea} '
                             'dim_hid:={n_hid} '
                             'activation:={act} '.format(
                                 n_fea=self.n_features,
                                 n_hid=dim_hid,
                                 act=activation))

    def __call__(self, rels, vecs):
        "Return a score of fact; closer to zero is better."
        batch_size = rels.shape[0]
        shape = (batch_size, -1)

        xp = self.xp
        if self.flag_unifw:
            rels = xp.zeros(rels.shape)

        bias = self.b(rels)
        weight = self.w(rels)

        X = self.g(self.f(vecs))

        return F.batch_matmul(X, weight, transa=True).reshape(shape) \
            + bias.reshape(shape)
