#!/usr/bin/env python

import numpy as np

class MultinomialNaiveBayes:
	def __init__(self, alpha=1.0):
		self.alpha = 1.0
		self.n_classes = -1


	def validate_params(self, X, y):
		return X.shape[0] == y.shape[0] and y.shape[1] == 1

	def fit_class_log_probs(self, y):
		""" compute p(class = c_i) """
		denom = np.log(y.shape[0])
		return np.log(self.cls_cnts) - denom

	def calc_feature_cnts_smoothed(self, X, y):
		result = np.zeros((self.num_features, self.n_classes))
		for label in range(self.n_classes):
			result[np.arange(self.num_features),label] = X[np.where(y==label)[0],:].sum(axis=0) + self.alpha
		return result

	def fit(self, X, y):
		"""
		X is assumed to be a NxW matrix, and y an Nx1 array containing class labels. (starting from 0 to n classes)

		result:
		W is (num_features, n_classes)
		b is (n_classes + 1)
		"""
		self.validate_params(X, y)

		self.num_features = X.shape[1]
		self.n_classes = np.max(y)+1
		self.cls_cnts = np.bincount(y.reshape(y.shape[0],))
		class_logp = self.fit_class_log_probs(y)
		self.feature_cnts = self.calc_feature_cnts_smoothed(X, y)
		# turn from shape (num_features,) to (num_features,1)
		denom = self.feature_cnts.sum(axis=1).reshape(-1, 1)
		self.feature_logp = np.log(self.feature_cnts) - np.log(denom)
		self.W = self.feature_logp
		self.b = class_logp

	def ll(self, X):
		# log( P(X|y) * P(y) )
		return np.dot(X, self.W) + self.b

	def predict(self, X):
		return np.argmax(self.ll(X), axis=1)

	def logsumexp(self, X, axis=1):
		X = np.rollaxis(X, axis)
		X_max = X.max(axis=0)
		result = np.log(np.sum(np.exp(X - X_max), axis=0))
		result += X_max
		return result

	def predict_proba(self, X):
		# log( P(X | Y) * P(Y))
		log_p_x_y = self.ll(X)
		# log(P(X))
		log_p_x = self.logsumexp(log_p_x_y, axis=1)
		# P( Y | X)
		log_probs = log_p_x_y - log_p_x.reshape((-1,1))
		return np.exp(log_probs)

