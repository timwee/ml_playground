#!/usr/bin/env python

from gradient_descent_optimizers import SGDOptimizer
import numpy as np
import scipy.sparse as sps
from sklearn.utils.extmath import safe_sparse_dot

class LogisticRegressionWithSGD:
	def __init__(self):
		pass

	def train(self, X, y, num_features, num_classes, num_iter=100, learning_rate=0.001, l2_reg_param=0.0, batch_size=64, decay=0.0):
		"""
		X is assumed to be (num_rows, num_features)
		y is assumed to be multiclass, values can be from [0, num_classes-1]
		"""
		sgd_optimizer = SGDOptimizer(num_iter=num_iter, batchsize=batch_size, lrate=learning_rate, reg=l2_reg_param, decay=decay)
		num_rows = X.shape[0]
		self.W = 0.01 * np.random.randn(num_features, num_classes)
		#self.b = np.zeros((1,num_classes))
		return sgd_optimizer.run(self.W, X, y, self.grad_loss_func)

	def train_with_opt(self, optimizer, X, y, num_features, num_classes):
		"""
		X is assumed to be (num_rows, num_features)
		y is assumed to be multiclass, values can be from [0, num_classes-1]
		"""
		num_rows = X.shape[0]
		self.W = 0.01 * np.random.randn(num_features, num_classes)
		#self.b = np.zeros((1,num_classes))
		return optimizer.run(self.W, X, y, self.grad_loss_func)


	def grad_loss_func(self, W, X, y, reg):
		# X.shape == (num_rows, num_features), W.shape == (num_features, num_classes)
		num_rows = X.shape[0]
		Z = safe_sparse_dot(X, W) 
		#+ b # (num_rows, num_classes)

		#M = np.max(Z, axis=1, keepdims=True)
		#denom_logsumexp = np.sum(np.exp(Z - M), axis=1, keepdims=True)
		#probs = np.exp(Z-M)/ denom_logsumexp

		exp_scores = np.exp(Z)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

		correct_logprobs = -np.log(probs[np.arange(num_rows),y])
        
		data_loss = np.sum(correct_logprobs)/num_rows # -log(p(y|x))
		reg_loss = 0.5*reg*np.sum(W*W)
		loss = data_loss + reg_loss
        
		grad = probs # (num_rows, num_classes)
		grad[np.arange(num_rows),y] -= 1
		grad /= num_rows
		dW = safe_sparse_dot(X.T, grad) # (num_features, num_classes)
		#db = np.sum(grad, axis=0, keepdims=True) # (1, num_classes)
		dW += reg*W
		return loss, dW
		#, db

	def predict_proba(self, X):
		#X_arr = X.toarray() if sps.issparse(X) else X

		Z = safe_sparse_dot(X, self.W) 
		#+ self.b # (num_rows, num_classes)
		exp_scores = np.exp(Z)
		probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
		return probs
		#M = np.max(Z, axis=1, keepdims=True)
		#denom_logsumexp = np.sum(np.exp(Z - M), axis=1, keepdims=True)
		#return np.exp(Z-M)/ denom_logsumexp
