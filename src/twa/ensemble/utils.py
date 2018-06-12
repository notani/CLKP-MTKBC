#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import numpy as np
import random


def init_logger(name='logger'):
    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_fmt = '%(asctime)s/%(name)s[%(levelname)s]: %(message)s'
    logging.basicConfig(format=log_fmt,
                        datefmt='%Y/%m/%d %H:%M:%S')
    return logger


def set_random_seed(seed, use_gpu=True):
    # set Python random seed
    random.seed(seed)

    # set NumPy random seed
    np.random.seed(seed)

    if use_gpu:
        # set Chainer(CuPy) random seed
        import cupy
        cupy.random.seed(seed)


def find_greatest_divisor(N, upper=5000):
    for i in reversed(range(max(upper, N // 2))):
        if N % i == 0:
            return i
    return 1


def standardize_vectors(vecs, mu, sigma):
    return (vecs - mu) / sigma
