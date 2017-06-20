#! /usr/bin/env python

import numpy as np
import math

EPS = 1e-8

class RMSProp:
	def __init__(self, lr=1e-3, rho=0.9, epsilon=EPS, decay=0.0, num_epochs=100, batchsize=64, l2_reg=1e-3):
		self.lr = lr
		self.rho = rho
		self.epsilon = epsilon
		self.decay = decay
		self.num_epochs = num_epochs
		self.batchsize = batchsize
		self.l2_reg = l2_reg

	def run(self, W, X, y, grad_loss_func):
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lr
		# the exponentially decaying sum of squares estimator
		sum_sq_est = np.zeros(W.shape)
		for it in range(self.num_epochs):
		    for offset in range(0, num_rows, self.batchsize):
		        minibatch = X[offset:min(num_rows, offset+self.batchsize),:] # (num_rows, num_features)
		        y_minibatch = y[offset:offset+self.batchsize]
		        loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.l2_reg)

		        sum_sq_est = self.rho * sum_sq_est + (1. - self.rho) * np.square(dW)
		        W -= ((lrate * dW) / np.sqrt(sum_sq_est + self.epsilon))
		        last_loss = loss
		    if self.decay > 0:
		     	lrate *= 1. / (1. + (self.decay * it))
		    print("iteration %d, loss %f" % (it, last_loss))

		return last_loss

############################################

class AdaGrad:
	def __init__(self, lr=1e-2, epsilon=EPS, decay=0.0, num_epochs=100, batchsize=64, l2_reg=1e-3):
		self.lr = lr
		self.epsilon = epsilon
		self.decay = decay
		self.num_epochs = num_epochs
		self.batchsize = batchsize
		self.l2_reg = l2_reg

	def run(self, W, X, y, grad_loss_func):
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lr
		# the exponentially decaying sum of squares estimator
		grad_sum_sq = np.zeros(W.shape)
		for it in range(self.num_epochs):
		    for offset in range(0, num_rows, self.batchsize):
		        minibatch = X[offset:min(num_rows, offset+self.batchsize),:] # (num_rows, num_features)
		        y_minibatch = y[offset:offset+self.batchsize]
		        loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.l2_reg)

		        grad_sum_sq += np.square(dW)
		        W -= ((lrate * dW) / np.sqrt(grad_sum_sq + self.epsilon))
		        last_loss = loss
		    if self.decay > 0:
		     	lrate *= 1. / (1. + (self.decay * it))
		    print("iteration %d, loss %f" % (it, last_loss))

		return last_loss

############################################

class AdaDelta:
	def __init__(self, lr=1.0, rho=0.95, epsilon=EPS, decay=0.0, num_epochs=100, batchsize=64, l2_reg=1e-3):
		self.lr = lr
		self.rho = rho
		self.epsilon = epsilon
		self.decay = decay
		self.num_epochs = num_epochs
		self.batchsize = batchsize
		self.l2_reg = l2_reg

	def run(self, W, X, y, grad_loss_func):
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lr
		# the exponentially decaying sum of squares estimator
		sum_sq_est = np.zeros(W.shape)
		delta_sum_sq_est = np.zeros(W.shape)
		for it in range(self.num_epochs):
		    for offset in range(0, num_rows, self.batchsize):
		        minibatch = X[offset:min(num_rows, offset+self.batchsize),:] # (num_rows, num_features)
		        y_minibatch = y[offset:offset+self.batchsize]
		        loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.l2_reg)

		        sum_sq_est = self.rho * sum_sq_est + (1. - self.rho) * np.square(dW)
		        update = dW * np.sqrt(delta_sum_sq_est + self.epsilon) / np.sqrt(sum_sq_est + self.epsilon)
		        W -= lrate * update
		        delta_sum_sq_est = self.rho * delta_sum_sq_est + (1 - self.rho) * np.square(update)
		        last_loss = loss
		    if self.decay > 0:
		     	lrate *= 1. / (1. + (self.decay * it))
		    print("iteration %d, loss %f" % (it, last_loss))

		return last_loss

############################################

class Adam:
	"""
	Default params from paper

	# Arguments:
		lr: float >= 0. Learning rate.
	    beta_1: float, 0 < beta < 1. Generally close to 1.
	    beta_2: float, 0 < beta < 1. Generally close to 1.
	    epsilon: float >= 0. Fuzz factor.
	- [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
	"""
	def __init__(self, lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=EPS, decay=0.0, num_epochs=100, batchsize=64, l2_reg=1e-3):
		self.lr = lr
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.decay = decay
		self.num_epochs = num_epochs
		self.batchsize = batchsize
		self.l2_reg = l2_reg

	def run(self, W, X, y, grad_loss_func):
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lr

		# first and second moments of gradients
		ms = np.zeros(W.shape)
		vs = np.zeros(W.shape)
		for it in range(self.num_epochs):
			# shortcut mentioned in paper
			lr_t = lrate * (math.sqrt(1. - math.pow(self.beta_2, it + 1)) / (1. - math.pow(self.beta_1, it + 1)))
			for offset in range(0, num_rows, self.batchsize):
				minibatch = X[offset:min(num_rows, offset+self.batchsize),:]
				y_minibatch = y[offset:offset+self.batchsize]
				loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.l2_reg)

				ms = (self.beta_1 * ms) + (1. - self.beta_1) * dW
				vs = (self.beta_2 * vs) + (1. - self.beta_2) * np.square(dW)
				W -= lr_t * (ms / np.sqrt(vs + self.epsilon))
				last_loss = loss
			if self.decay > 0:
				lrate *= 1. / (1. + (self.decay * it))
			print("iteration %d, loss %f" % (it, last_loss))

		return last_loss

############################################

class NAdam:
	"""
	Nesterov Adam

	# Arguments:
        lr >= 0. Learning rate.
        beta params similar to adam
	1. [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
    2. [On the importance of initialization and momentum in deep learning](http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
	"""
	def __init__(self, lr=2e-3, beta_1=0.9, beta_2=0.999, epsilon=EPS, decay=4e-3, num_epochs=100, batchsize=64, l2_reg=1e-3):
		self.lr = lr
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		self.epsilon = epsilon
		self.decay = decay
		self.num_epochs = num_epochs
		self.batchsize = batchsize
		self.l2_reg = l2_reg
		self.m_schedule = 1.0

	def run(self, W, X, y, grad_loss_func):
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lr

		# first and second moments of gradients
		ms = np.zeros(W.shape)
		vs = np.zeros(W.shape)
		for it in range(self.num_epochs):
			# shortcut mentioned in paper
			# Due to the recommendations in [2], i.e. warming momentum schedule
			t = it + 1
			momentum_cache_t = self.beta_1 * (1. - 0.5 * (math.pow(0.96, t * self.decay)))
			momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (math.pow(0.96, (t + 1) * self.decay)))
			m_schedule_new = self.m_schedule * momentum_cache_t
			m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1

			#lr_t = lrate * (math.sqrt(1. - math.pow(self.beta_2, it + 1)) / (1. - math.pow(self.beta_1, it + 1)))
			for offset in range(0, num_rows, self.batchsize):
				minibatch = X[offset:min(num_rows, offset+self.batchsize),:]
				y_minibatch = y[offset:offset+self.batchsize]
				loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.l2_reg)

				g_prime = dW / (1. - m_schedule_new)
				ms = self.beta_1 * ms + (1. - self.beta_1) * dW
				m_t_unbiased = ms / (1. - m_schedule_next)

				vs = self.beta_2 * vs + (1. - self.beta_2) * np.square(dW)
				v_t_unbiased = vs / (1. - math.pow(self.beta_2, t))

				m_t_bar = (1. - momentum_cache_t) * g_prime + momentum_cache_t_1 * m_t_unbiased

				W -= lrate * (m_t_bar / (np.sqrt(v_t_unbiased) + self.epsilon))
				last_loss = loss
			print("iteration %d, loss %f" % (it, last_loss))

		return last_loss

############################################