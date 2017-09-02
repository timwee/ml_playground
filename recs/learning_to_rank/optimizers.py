#!/usr/bin/env python

import numpy as np
from collections import defaultdict as dd
import math

GRAD_MAX = 100


class SGD:
    def __init__(self, lr=1e-5, decay=0.0, momentum=0.0, nesterov=False):
        self.initial_lr = lr
        self.lr = lr
        self.decay = decay
        self.mu = momentum
        self.nesterov = nesterov
        self.grad_max = GRAD_MAX # clip gradient if > than this value
        self.momentum_cache = {}

    def reset(self):
        self.momentum_cache = {}

    def _get_decayed_lr(self, num_iter):
        return self.lr * (1. / (1. + self.decay * num_iter))

    def __call__(self, weights, grads, idxs, cur_iter, param_name):
        lr = self._get_decayed_lr(cur_iter)

        clipped_gradient = np.clip(grads, -GRAD_MAX, GRAD_MAX)
        if self.mu == 0.0:
            weights[idxs] -= lr * clipped_gradient
        else:
            if param_name not in self.momentum_cache:
                self.momentum_cache[param_name] = np.zeros_like(weights)
            momentum = self.momentum_cache[param_name]
            if self.nesterov:
                # http://cs231n.github.io/neural-networks-3/#sgd
                ## we would like to do the following
                # x_ahead = x + mu * v
                # evaluate dx_ahead (the gradient at x_ahead instead of at x)
                # v = mu * v - learning_rate * dx_ahead
                # x += v
                ### in practice, people repfer to express it similar to SGD
                # v_prev = v # back this up
                # v = mu * v - learning_rate * dx # velocity update stays the same
                # x += -mu * v_prev + (1 + mu) * v # position update changes form
                weights[idxs] -= self.mu * momentum[idxs]

                momentum[idxs] *= self.mu 
                momentum[idxs] -= lr * clipped_gradient
                weights[idxs] += (1. + self.mu) * momentum[idxs]
                
            else:
                # http://cs231n.github.io/neural-networks-3/#sgd
                # Momentum update
                # v = mu * v - learning_rate * dx # integrate velocity
                # x += v # integrate position
                momentum[idxs] *= self.mu
                momentum[idxs] -= lr * clipped_gradient
                weights[idxs] += momentum[idxs]
        return lr
                
class NAdam:
    """
    Nesterov Adam

    # Arguments:
        lr >= 0. Learning rate.
        beta params similar to adam
    1. [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
    2. [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """
    def __init__(self, lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=4e-3):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.m_schedule = 1.0
        self.moments1 = {}
        self.moments2 = {}

    def reset(self):
        self.moments1 = {}
        self.moments2 = {}

    def _get_or_init_cache(self, cache, weights, k):
        if k not in cache:
            cache[k] = np.zeros_like(weights)
        return cache[k]

    def __call__(self, weights, grads, idxs, cur_iter, param_name):
        t = cur_iter + 1
        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (math.pow(0.96, t * self.decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (math.pow(0.96, (t + 1) * self.decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1

        # first moment
        ms = self._get_or_init_cache(self.moments1, weights, param_name)
        # second moment
        vs = self._get_or_init_cache(self.moments2, weights, param_name)

        g_prime = grads / (1. - m_schedule_new)

        # first moment unbiased
        ms[idxs] *= self.beta_1
        ms[idxs] += (1. - self.beta_1) * grads
        m_t_unbiased = ms[idxs] / (1. - m_schedule_next)

        #2nd moment unbiased
        vs[idxs] *= self.beta_2
        vs[idxs] += (1. - self.beta_2) * np.square(grads)
        v_t_unbiased = vs[idxs] / (1. - math.pow(self.beta_2, t))

        m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_unbiased

        #lr = lrate * (math.sqrt(1. - math.pow(self.beta_2, it + 1)) / (1. - math.pow(self.beta_1, it + 1)))
        lr = self.lr
        weights[idxs] -= lr * (m_t_bar / (np.sqrt(v_t_unbiased) + self.epsilon))

        return lr



