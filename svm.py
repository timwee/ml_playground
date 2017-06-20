#! /usr/bin/env python

from gradient_descent_optimizers import SGDOptimizer
import numpy as np
import math
from sklearn.utils.extmath import safe_sparse_dot

import scipy.sparse as sps

class LinearSVM:
	def __init__(self):
		pass

	def train(self, X, y, num_features, num_classes, margin=1, num_iter=100, decay=0.0, learning_rate=0.001, l2_reg_param=0.0, batch_size=64):
		"""
		X is assumed to be (num_rows, num_features)
		y is assumed to be multiclass, values can be from [0, num_classes-1]
		"""
		sgd_optimizer = SGDOptimizer(num_iter=num_iter, batchsize=batch_size, lrate=learning_rate, reg=l2_reg_param, decay=decay)
		num_rows = X.shape[0]
		self.W = 0.01 * np.random.randn(num_features, num_classes)
		self.margin = margin
		self.num_classes = num_classes
		self.num_features = num_features
		#self.b = np.zeros((1,num_classes))
		return sgd_optimizer.run(self.W, X, y, self.grad_loss_func)
		#return sgd_optimizer.run_no_intercept(self.W, X, y, self.grad_loss_func)

	def train_with_opt(self, optimizer, X, y, num_features, num_classes, margin=1):
		"""
		X is assumed to be (num_rows, num_features)
		y is assumed to be multiclass, values can be from [0, num_classes-1]
		"""
		num_rows = X.shape[0]
		self.W = 0.01 * np.random.randn(num_features, num_classes)
		self.margin = margin
		self.num_classes = num_classes
		self.num_features = num_features
		#self.b = np.zeros((1,num_classes))
		return optimizer.run(self.W, X, y, self.grad_loss_func)

	def grad_loss_func2(self, W, X, y, reg):
		"""
		Structured SVM loss function, vectorized implementation.

		Inputs and outputs are the same as svm_loss_naive.
		"""
		# X.shape == (num_rows, num_features), W.shape == (num_features, num_classes)
		loss = 0.0
		num_train = X.shape[0]
		num_features = W.shape[0]
		num_classes = W.shape[1]
		dW = np.zeros(W.shape) # initialize the gradient as zero  

		############################################################################# 
		Z = safe_sparse_dot(X, W)
		Loss = Z - (Z[np.arange(num_train), y]+1).reshape((-1,1))
		Bool = Loss > 0
		assert Bool.shape == (num_train, num_classes)
		# On sum sur les colonnes et on enlceve la valeur que l'on a compteur en trop
		Loss = np.sum(Loss * Bool , axis = 0) - 1.0
		Regularization = 0.5 * reg * np.sum(W*W)
		loss = np.sum(Loss) / float(num_train) +Regularization
		#############################################################################
		Bool = Bool*np.ones(Loss.shape)
		Bool[[np.arange(num_train), y]] = -(np.sum(Bool,axis=1)-1)
		dW = safe_sparse_dot(X.T, Bool) / float(num_train)
		assert dW.shape == (num_features, num_classes)
		dW += reg * W
		return loss, dW


	def grad_loss_func(self, W, X, y, reg):
		# X.shape == (num_rows, num_features), W.shape == (num_features, num_classes)
		num_classes = W.shape[1]
		num_features = X.shape[1]
		
		minibatch = X # (num_rows, num_features)
		if sps.issparse(minibatch):
			minibatch = minibatch.toarray()
		num_rows = minibatch.shape[0]
		y_minibatch = y
		Z = safe_sparse_dot(minibatch, W) 
		#+ b # (num_rows, num_classes)

		# correct class's predicted value
		s_yi = Z[np.arange(num_rows), y_minibatch]
		margins = (Z - s_yi.reshape((-1,1))) + self.margin
		#assert margins.shape == (num_rows, num_classes)
		elem_loss = np.maximum(0, margins)
		#assert elem_loss.shape == margins.shape
		elem_loss[np.arange(num_rows), y_minibatch] = 0
		data_loss = np.sum(elem_loss) / num_rows
		reg_loss = 0.5 * reg *  math.pow(np.linalg.norm(W, 2), 2)
		#0.5 * reg * np.sum(W*W)
		loss = data_loss + reg_loss

		#assert elem_loss.shape == (num_rows, num_classes)
		       
		# dW - for incorrect classes
		# minibatch.shape == (num_rows, num_features), elem_loss.shape ==(num_rows, num_classes)
		# dW.shape == (num_features, num_classes)
		dW = safe_sparse_dot(minibatch.T, (elem_loss > 0)) / num_rows
		# to compute adjustment for correct class:
		#   the adjustment for each train sample's feature is -(num_violated_margin * feature_val)
		#   mask out the correct class's X values and the ones that don't cross the margin
		#   sum on each

		# count # of classes violating margin per sample
		num_violated = np.sum((elem_loss > 0), axis=1)
		assert num_violated.shape == (num_rows,)
		# multiply by feature_val of correct class
		# minibatch.shape == (num_rows, num_features), num_violated.
		#assert minibatch.shape == (num_rows, num_features)
		#assert num_violated.shape == (num_rows,)
		correct_class_adj = minibatch.T * num_violated
		#print(correct_class_adj.shape)
		#assert correct_class_adj.shape == (num_features, num_rows)

		# correct_class_adj.shape == (num_features, num_rows), 

		#dW -= np.dot(correct_class_adj, (elem_loss == 0))
		dW -= safe_sparse_dot(correct_class_adj, (elem_loss == 0)) / num_rows

		#  correct_cls_oj = np.array([scores[correct_class, train_num] for train_num, correct_class in enumerate(y)])
		#assert dW.shape == W.shape
		#dW /= float(num_rows)
		dW += (reg * W)
		return loss, dW

	def predict_proba(self, X):
		scores = safe_sparse_dot(X, self.W)
		return scores
		#M = np.max(Z, axis=1, keepdims=True)
		#denom_logsumexp = np.sum(np.exp(Z - M), axis=1, keepdims=True)
		#return np.exp(Z-M)/ denom_logsumexp
