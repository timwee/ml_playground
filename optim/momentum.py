#! /usr/bin/env python
import numpy as np

PRINT_INTERVAL = 10

class SGDWithMomentum:
	def __init__(self, lr=0.01, momentum=0.0, decay=0.0, nesterov=False, num_epochs=100, batchsize=64, l2_reg=1e-3):
		self.lr = lr
		self.momentum = momentum
		self.decay = decay
		self.nesterov = nesterov
		self.num_epochs = num_epochs
		self.batchsize = batchsize
		self.l2_reg = l2_reg

	def run(self, W, X, y, grad_loss_func):
		num_rows = X.shape[0]
		last_loss = 0
		lrate = self.lr
		m = np.zeros(W.shape)
		for it in range(self.num_epochs):
		    for offset in range(0, num_rows, self.batchsize):
		        minibatch = X[offset:min(num_rows, offset+self.batchsize),:] # (num_rows, num_features)
		        y_minibatch = y[offset:offset+self.batchsize]
		        loss, dW = grad_loss_func(W, minibatch, y_minibatch, self.l2_reg)

		        # using Dozat's notation
		        # velocity
		        v = self.momentum * m + lrate * dW
		        m = v
		        if self.nesterov:
		        	W -= (self.momentum * v + lrate * dW)
		        else:
		        	W -= v
		        #loss_history.append(loss)
		        last_loss = loss
		    lrate *= 1. / (1. + (self.decay * it))
		    if it % PRINT_INTERVAL == 0:
		    	print("iteration %d, loss %f" % (it, last_loss))

		return last_loss