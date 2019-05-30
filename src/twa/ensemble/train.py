#!/usr/bin/env python
# -*- coding: utf-8 -*-

from chainer import Chain, Variable, cuda, optimizers, training
from chainer import functions as F
from chainer.dataset import iterator as iterator_module
from chainer.training import extensions
from os import path
import argparse
import chainer
import copy
import numpy as np

from dataset import COL_BASIC_FEATURES
from dataset import FactIterator, Vocabulary, read_dataset, get_idx2vec, get_values, replace_by_dic
from models import LinearEnsembler, MLPEnsembler
from utils import init_logger
from utils import set_random_seed
from utils import standardize_vectors
from utils import find_greatest_divisor

verbose = False
logger = init_logger('Ensember')

# Directories
dir_scripts = path.dirname(path.dirname(path.dirname(path.abspath(__file__))))
dir_root = path.dirname(dir_scripts)
dir_data = path.join(dir_root, 'data')


class Classifier(Chain):
    """Calculate loss."""

    def __init__(self, predictor, label2fact, en2ja, idx2vec,
                 margin=1.0, lam=None,
                 use_gpu=False):
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
        self.margin = margin
        self.lam = lam  # L2 penalty
        assert self.lam is None or self.lam >= 0, '--lam must be >= 0'
        super(Classifier, self).__init__(predictor=predictor)

    def to_cpu(self):
        # for k, v in self.ja_cands.items():
        #     self.ja_cands[k] = cuda.to_cpu(v)
        for k, v in self.en2ja.items():
            self.en2ja[k] = cuda.to_cpu(v)
        super(Classifier, self).to_cpu()

    def to_gpu(self, device=None):
        # for k, v in self.ja_cands.items():
        #     self.ja_cands[k] = cuda.to_gpu(v)
        for k, v in self.en2ja.items():
            self.en2ja[k] = cuda.to_gpu(v)
        super(Classifier, self).to_gpu(device=device)

    def __call__(self, rels, vecs, en_indices):
        """Run the predictor and return a loss value"""
        loss = self.calc_loss(rels, vecs, en_indices)
        chainer.report({'loss': loss}, self)

        return loss

    def calc_loss(self, rels, vecs, en_indices):
        """Return loss"""

        def choice_negative_sample(idx):
            return np.random.choice(self.ja_cands[idx])

        score_pos = self.predictor(rels, vecs)
        xp = cuda.get_array_module(score_pos)

        # Draw a negative example
        indices = xp.array([choice_negative_sample(f) for f in en_indices])
        vecs_neg = self.idx2vec[indices]
        score_neg = self.predictor(rels, vecs_neg)

        margin = xp.zeros(score_pos.shape, dtype=xp.float32)
        margin[:] = self.margin
        margin_vecs = Variable(margin)
        loss_ind = F.relu(margin_vecs - score_pos + score_neg)
        loss = F.average(loss_ind)

        # Regularization
        if self.lam:
            loss += self.lam * F.sum(F.square(self.predictor.theta))

        return loss

    def calc_ranks(self, rels, en_indices):
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
            buff['idx'].append(indices[:])
            buff['vec'].append(self.idx2vec[indices][:])
            rels_ = xp.empty(self.idx2vec[indices].shape[0],
                             dtype=xp.int32)
            rels_[:] = rel
            buff['rel'].append(rels_[:])
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

        ranks = xp.array(ranks)

        return {'meanrank': sum(ranks) / len(en_indices),
                'mrr': sum(1/ranks) / len(en_indices)}


class Updater(training.StandardUpdater):
    """Update network parameters."""

    def __init__(self, train_iter, optimizer, device):
        """Construct a Updater.

        Args:
            train_iter: Training data which is iterable
            optimizer: chainer.optimizers object
        """
        super(Updater, self).__init__(
            train_iter,
            optimizer,
            device=device
        )

    def update_core(self):
        """Update target parameters."""
        train_iter = self.get_iterator('main')
        optimizer = self.get_optimizer('main')

        rels, vecs, en_indices = train_iter.__next__()
        if self.device >= 0:
            rels = cuda.to_gpu(rels)
            vecs = cuda.to_gpu(vecs)
            # en_indices = cuda.to_gpu(en_indices)
        loss = optimizer.target(rels, vecs, en_indices)

        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()


class Evaluator(extensions.Evaluator):
    """Evaluate a model on a validation set."""

    def __init__(self, iterator, target, converter=None,
                 device=None, eval_hook=None, eval_func=None):
        """Construct a Evaluator.

        Args:
            iterator: Evaluation data which is iterable
            target: Classifier object
        """
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        if isinstance(target, chainer.link.Link):
            target = {'main': target}
        self._targets = target

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self.eval_func = eval_func

        self.scores = []

    def evaluate(self):
        """Report evaluation for the model."""
        iterator = self._iterators['main']
        target = self._targets['main']

        it = copy.copy(iterator)
        summary = chainer.reporter.DictSummary()
        for x in it:
            observation = {}
            rels, vecs, en_indices = x
            if self.device >= 0:
                rels = cuda.to_gpu(rels)
                vecs = cuda.to_gpu(vecs)
                # en_indices = cuda.to_gpu(en_indices)
            with chainer.reporter.report_scope(observation):
                target(rels, vecs, en_indices)
                d = target.calc_ranks(rels, en_indices)
                chainer.report(d, target)
            summary.add(observation)

        self.scores.append(summary.compute_mean())  # Save scores
        return summary.compute_mean()

    def get_best_score(self, col='validation/main/mrr', ascending=False):
        idx = 0 if ascending else -1
        return sorted(
            [(epoch, record[col])
             for epoch, record in enumerate(self.scores, start=1)],
            key=lambda t: t[1])[idx]


def save_best_model(dir_out, model,
                    col='validation/main/mrr', ascending=False,
                    mu=None, sigma=None):
    import json
    import os
    with open(os.path.join(dir_out, 'log')) as f:
        d = json.load(f)

    idx = 0 if ascending else -1
    best = sorted(d, key=lambda t: t[col])[idx]

    path_model = os.path.join(dir_out, 'model_iter_' + str(best['iteration']))
    path_model_new = os.path.join(dir_out, 'ensemble.model')
    chainer.serializers.load_npz(path_model, model)
    model.add_persistent('_mu', mu)
    model.add_persistent('_sigma', sigma)
    chainer.serializers.save_npz(path_model_new, model)

    # Remove unused models
    for filepath in os.listdir(dir_out):
        if filepath.startswith('model_iter') \
           or filepath.startswith('snapshot_iter'):
            os.remove(os.path.join(dir_out, filepath))


def report_params(params):
    logger.info('learning rate: {}'.format(params.lr))
    logger.info('batch size: {}'.format(params.batch_size))
    logger.info('polynomial degree: {}'.format(params.poly_degree))
    logger.info('standardize vectors: {}'.format(params.flag_standardize))
    logger.info('save a model: {}'.format(params.save))


def main(args):
    global verbose
    verbose = args.verbose

    assert args.poly_degree >= 1, '--degree must be positive integer'
    poly_degree = args.poly_degree

    if verbose:
        report_params(args)

    gpu = args.gpu
    if gpu >= 0:
        cuda.check_cuda_available()
        if verbose:
            logger.info('Use GPU {}'.format(gpu))
        cuda.get_device_from_id(gpu).use()

    set_random_seed(0, use_gpu=(gpu >= 0))

    n_epochs = args.n_epochs
    batch_size = args.batch_size

    # Dataset
    dfs = {}
    dfs['train'] = read_dataset(path.join(args.dir_in, args.filename_train))
    dfs['devel'] = read_dataset(path.join(args.dir_in, args.filename_devel))

    # Load relation vocabulary
    rel2id = Vocabulary()
    rel2id.read_from_file(args.path_rels)

    # Load concept vocabulary
    if verbose:
        logger.info('Load vocabulary')
    fact2id = Vocabulary()
    fact2id.read_from_list(np.unique(get_values(list(dfs.values()), 'fact')))
    ja2id = Vocabulary()
    ja2id.read_from_list(np.unique(get_values(list(dfs.values()), 'fact_ja')))
    en2id = Vocabulary()
    en2id.read_from_list(np.unique(get_values(list(dfs.values()), 'fact_en')))

    if verbose:
        logger.info('Replace facts with indices')
    for col in dfs.keys():
        dfs[col].loc[:, 'fact'] = replace_by_dic(dfs[col]['fact'], fact2id).astype(np.int32)
        dfs[col].loc[:, 'fact_ja'] = replace_by_dic(dfs[col]['fact_ja'], ja2id).astype(np.int32)
        dfs[col].loc[:, 'fact_en'] = replace_by_dic(dfs[col]['fact_en'], en2id).astype(np.int32)
        dfs[col].loc[:, 'rel'] = replace_by_dic(dfs[col]['rel'], rel2id).astype(np.int32)
    label2fact = {i: set(np.concatenate([df[df['twa'] == i]['fact'].unique()
                                         for df in dfs.values()]))
                  for i in [0, 1]}
    en2ja = {en: set(df[df['fact_en'] == en]['fact'].unique())
             for df in dfs.values()
             for en in sorted(df['fact_en'].unique())}
    idx2vec = get_idx2vec(list(dfs.values()), poly_degree=poly_degree)

    n_facts = len(fact2id)
    n_en = len(en2id)
    n_ja = len(ja2id)
    assert n_facts+1 == len(idx2vec), '{}[n_facts] != {}[len(idx2vec)]'.format(
        n_facts+1, len(idx2vec))

    if verbose:
        logger.info('Alignment: {}'.format(n_facts))
        logger.info('En: {}'.format(n_en))
        logger.info('Ja: {}'.format(n_ja))
        logger.info('Train: {}'.format(len(dfs['train'])))
        logger.info('Devel: {}'.format(len(dfs['devel'])))

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

    # Set up a dataset iterator
    train_iter = FactIterator(dfs['train'], args.batch_size, ja2id, en2id,
                              train=True, repeat=True,
                              poly_degree=poly_degree)
    # Only keep positive examples in development set
    df = dfs['devel'][dfs['devel']['twa'] == 1].drop_duplicates('fact_en')

    # Set batch size
    batch_size = find_greatest_divisor(len(df))
    if batch_size == 1 and len(df) <= 10**4:
        batch_size = len(df)
    if verbose:
        logger.info('Devel batch size = {}'.format(batch_size))
    devel_iter = FactIterator(df, batch_size, ja2id, en2id,
                              train=False, repeat=False,
                              poly_degree=poly_degree)

    # Standardize vectors
    if args.flag_standardize:
        mu, sigma = train_iter.standardize_vectors()
        devel_iter.standardize_vectors(mu=mu, sigma=sigma)
        idx2vec = standardize_vectors(idx2vec, mu, sigma)
    else:
        mu, sigma = None, None

    if gpu >= 0:
        idx2vec = cuda.to_gpu(idx2vec)

    # Set up a model
    model = Classifier(ensembler, label2fact, en2ja, idx2vec,
                       margin=args.margin, lam=args.lam)

    if gpu >= 0:
        model.to_gpu(device=gpu)

    # Set up an optimizer
    optimizer = optimizers.AdaGrad(lr=args.lr)
    optimizer.setup(model)

    # Set up a trainer
    updater = Updater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (n_epochs, 'epoch'), out=args.dir_out)

    # evaluate development set
    evaluator = Evaluator(devel_iter, model, device=gpu)
    trainer.extend(evaluator)

    # Write out a log
    trainer.extend(extensions.LogReport())
    # Display training status
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'validation/main/meanrank', 'validation/main/mrr', 'elapsed_time']))

    if args.save:
        trainer.extend(extensions.snapshot(), trigger=(args.n_epochs, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            ensembler, 'model_iter_{.updater.iteration}'),
                       trigger=(1, 'epoch'))

    # Launch training process
    trainer.run()

    # Report the best score
    (epoch, score) = evaluator.get_best_score()
    if verbose:
        logger.info('Best score: {} (epoch={})'.format(score, epoch))

    # Clean the output directory
    if args.save:
        save_best_model(args.dir_out, ensembler,
                        mu=mu, sigma=sigma)

    del dfs
    del fact2id
    del ja2id
    del en2id

    return score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    path_rels = path.join(dir_data, 'rel.txt')

    parser.add_argument('-i', '--input', dest='dir_in',
                        default='.',
                        help='input directory')
    parser.add_argument('--train', dest='filename_train',
                        default='train.tsv')
    parser.add_argument('--devel', dest='filename_devel',
                        default='devel.tsv')
    parser.add_argument('-o', '--output', dest='dir_out',
                        default='result',
                        help='output directory')
    parser.add_argument('-m', '--model', dest='model',
                        default='linear', help='model architecture')
    parser.add_argument('-s', '--save', action='store_true', default=False,
                        help='save snapshot')
    parser.add_argument('-b', '--batchsize', dest='batch_size',
                        type=int, default=int(64),
                        help='learning minibatch size')
    parser.add_argument('-e', '--epoch', dest='n_epochs',
                        type=int, default=int(30),
                        help='number of epochs to learn')
    parser.add_argument('--lr', type=float, default=float(0.1),
                        help='learning rate')
    parser.add_argument('--lam', type=float, help='L2 penalty')
    parser.add_argument('--path-rels', dest='path_rels',
                        default=path_rels)
    parser.add_argument('--margin', dest='margin', default=1.0, type=float,
                        help='margin in a loss function')
    parser.add_argument('--degree', dest='poly_degree', default=1, type=int,
                        help='degree of polynomial transformation')
    parser.add_argument('-g', '--gpu', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--uniform-weight', dest='flag_unifw',
                        action='store_true', default=False,
                        help='use uniform weight over all rels'
                        'in terms of meanrank on the devel set')
    parser.add_argument('--standardize', dest='flag_standardize',
                        action='store_true', help='standardize vectors')
    parser.add_argument('-v', '--verbose',
                        action='store_true', default=False,
                        help='verbose output')
    args = parser.parse_args()
    main(args)
