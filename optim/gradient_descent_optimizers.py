#!/usr/bin/env python

import scipy.sparse as sps

DECAY = 0.8
MIN_LRATE = 1e-4
PRINT_INTERVAL = 10

class SGDOptimizer:
	def __init__(self, num_iter=100, batchsize=64, lrate=1e-2, reg=1e-3, decay=0.0):
		self.num_iter = num_iter
		self.batchsize = batchsize
		self.lrate = lrate
		self.reg = reg
		self.decay = decay

	def run_no_intercept(self, W, X, y, grad_loss_func):
		# TODO: change to take in params instead of assuming a weight matrix
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lrate
		for it in range(self.num_iter):
		    for offset in range(0, num_rows, self.batchsize):
		        minibatch = X[offset:min(num_rows, offset+self.batchsize),:] # (num_rows, num_features)
		        #if sps.issparse(minibatch):
		        #	minibatch = minibatch.toarray()
		        y_minibatch = y[offset:offset+self.batchsize]
		        loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.reg)
		        #loss_history.append(loss)
		        last_loss = loss
		        W += -lrate * dW
		    lrate *= 1. / (1. + (self.decay * it))
		    if it % PRINT_INTERVAL == 0:
		    	print("iteration %d, loss %f" % (it, last_loss))
		    #if (it % 10) == 0 and it > 0 and self.lr_sched:
		    	#print("lessening learning rate from %f to %f" % (lrate, lrate * DECAY))
		    	#lrate *= DECAY
		    	#lrate = max(MIN_LRATE, lrate)
		return last_loss

	def run(self, W, X, y, grad_loss_func):
		# TODO: change to take in params instead of assuming a weight matrix
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lrate
		for it in range(self.num_iter):
		    for offset in range(0, num_rows, self.batchsize):
		        minibatch = X[offset:min(num_rows, offset+self.batchsize),:] # (num_rows, num_features)
		        #if sps.issparse(minibatch):
		        #	minibatch = minibatch.toarray()
		        y_minibatch = y[offset:offset+self.batchsize]
		        loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.reg)
		        #loss, dW, db = grad_loss_func(W, b, minibatch, y_minibatch, self.reg)
		        #loss_history.append(loss)
		        last_loss = loss
		        W += -lrate * dW
		    lrate *= 1. / (1. + (self.decay * it))
		    if it % PRINT_INTERVAL == 0:
		    	print("iteration %d, loss %f" % (it, last_loss))
		    # if (it % 10) == 0 and it > 0 and self.lr_sched:
		    # 	print("lessening learning rate from %f to %f" % (lrate, lrate * DECAY))
		    # 	lrate *= DECAY
		    # 	lrate = max(MIN_LRATE, lrate)
		return last_loss