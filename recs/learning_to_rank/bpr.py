#!/usr/bin/env python

from scipy.sparse import coo_matrix, csr_matrix, diags
import numpy as np
import math
from sklearn.utils.extmath import safe_sparse_dot
from base import LTRBase, nnz_csrrow

def sigmoid(x):
    # can also use scipy.special.expit
    return 1 / (1 + np.exp(-x))

class BPR(LTRBase):
    """
    https://arxiv.org/abs/1205.2618

    BPR: Bayesian Personalized Ranking from Implicit Feedback
    Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme
    """
    def _train_epoch(self, train):
        num_users, num_items = train.shape
        shuffled_users = np.arange(num_users)
        self.random_state.shuffle(shuffled_users)
        for u in range(num_users):
            user_id = shuffled_users[u]
            # get user items
            pos_item_idxs, pos_item_ratings = nnz_csrrow(train, user_id)
            pos_item_set = set(pos_item_idxs)
            shuffled_pos_items = np.arange(pos_item_idxs.shape[0])
            self.random_state.shuffle(shuffled_pos_items)
            for pos_item_id in shuffled_pos_items:
                pos_item_score = self._predict(user_id, pos_item_id)
                # sample until we find a negative item. Uniform sampling
                num_sampled = 0
                while num_sampled < self.max_samples:
                    num_sampled += 1
                    neg_item_id = self.random_state.randint(0, num_items)
                    if neg_item_id in pos_item_set:
                        continue
                    neg_item_score = self._predict(user_id, neg_item_id)
                    # found a candidate negative item
                    loss =  (1.0 - sigmoid(pos_item_score - neg_item_score))
                    # update biases
                    self._update_bias_grad(self.user_biases, user_id, loss, self.user_reg)
                    self._update_bias_grad(self.item_biases, neg_item_id, loss, self.item_reg)
                    self._update_bias_grad(self.item_biases, pos_item_id, -loss, self.item_reg)
                    
                    user_embedding = self.user_embeddings[user_id,:]
                    pos_item_embedding = self.item_embeddings[pos_item_id,:]
                    neg_item_embedding = self.item_embeddings[neg_item_id,:]
                    # update latent factors
                    self._update_latent_vector_grad(self.user_embeddings, \
                                              user_id, \
                                              loss * (neg_item_embedding - pos_item_embedding), \
                                              self.user_reg)
                    self._update_latent_vector_grad(self.item_embeddings, \
                                              pos_item_id, \
                                              -loss * user_embedding, \
                                              self.item_reg)
                    self._update_latent_vector_grad(self.item_embeddings, \
                                              neg_item_id, \
                                              loss * user_embedding, \
                                              self.item_reg)
                    self.user_scale *= (1.0 + self.user_reg * self.lr)
                    self.item_scale *= (1.0 + self.item_reg * self.lr)
                    break
                if self.user_scale > self.MAX_REG_SCALE or self.item_scale >self.MAX_REG_SCALE:
                    #print("regularizing!")
                    self._regularize()
        self._regularize()