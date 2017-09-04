#!/usr/bin/env python

from scipy.sparse import coo_matrix, csr_matrix, diags
import numpy as np
import math
from sklearn.utils.extmath import safe_sparse_dot
from base import LTRBase, nnz_csrrow

class WARPNaive(LTRBase):
    """
    WSABIE: Scaling Up To Large Vocabulary Image Annotation 
    http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf
    """
        
    def _train_epoch(self, train, cur_iter):
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
                # sample until we find a negative item that violates the margin
                num_sampled = 0
                while num_sampled < self.max_samples:
                    num_sampled += 1
                    neg_item_id = self.random_state.randint(0, num_items)
                    if neg_item_id in pos_item_set:
                        continue
                    neg_item_score = self._predict(user_id, neg_item_id)
                    if neg_item_score <= pos_item_score - 1:
                        continue
                    # found a candidate negative item
                    loss = math.log(math.floor(num_items / num_sampled))
                    if loss > self.MAX_LOSS:
                        loss = self.MAX_LOSS

                    avg_lr = 0.0
                    # update biases
                    avg_lr += self._update_bias_grad(self.user_biases, user_id, loss, self.user_reg, \
                        item_key="user_bias", cur_iter=cur_iter)
                    avg_lr += self._update_bias_grad(self.item_biases, neg_item_id, loss, self.item_reg, \
                        item_key="item_bias", cur_iter=cur_iter)
                    avg_lr += self._update_bias_grad(self.item_biases, pos_item_id, -loss, self.item_reg, \
                        item_key="item_bias", cur_iter=cur_iter)
                    
                    user_embedding = self.user_embeddings[user_id,:]
                    pos_item_embedding = self.item_embeddings[pos_item_id,:]
                    neg_item_embedding = self.item_embeddings[neg_item_id,:]
                    # update latent factors
                    avg_lr += self._update_latent_vector_grad(self.user_embeddings, \
                                              user_id, \
                                              loss * (neg_item_embedding - pos_item_embedding) + self.user_reg * user_embedding, \
                                              self.user_reg,
                                              item_key="user_embeddings", \
                                              cur_iter=cur_iter)
                    avg_lr += self._update_latent_vector_grad(self.item_embeddings, \
                                              pos_item_id, \
                                              -loss * user_embedding + self.item_reg * pos_item_embedding, \
                                              self.item_reg,
                                              item_key="item_embeddings", \
                                              cur_iter=cur_iter)
                    avg_lr += self._update_latent_vector_grad(self.item_embeddings, \
                                              neg_item_id, \
                                              loss * user_embedding + self.item_reg * neg_item_embedding, \
                                              self.item_reg,
                                              item_key="item_embeddings", \
                                              cur_iter=cur_iter)
                    avg_lr /= (3. + (self.num_factors * 3))
                    #self.user_scale *= (1.0 + self.user_reg * avg_lr)
                    #self.item_scale *= (1.0 + self.item_reg * avg_lr)
                    break
                if self.user_scale > self.MAX_REG_SCALE or self.item_scale > self.MAX_REG_SCALE:
                    #print("regularizing!")
                    #self._regularize()
                    pass
        #self._regularize()

class KosWARP(LTRBase):
    """
    Learning to Rank Recommendations with the k-Order Statistic Loss 
    https://research.google.com/pubs/pub41534.html
    """
        
    def _train_epoch(self, train, cur_iter):
        MAX_LOSS = 10
        num_users, num_items = train.shape
        shuffled_users = np.arange(num_users)
        self.random_state.shuffle(shuffled_users)
        for u in range(num_users):
            user_id = shuffled_users[u]
            # get user items
            pos_item_idxs, pos_item_ratings = nnz_csrrow(train, user_id)
            pos_item_set = set(pos_item_idxs)
            for i in pos_item_idxs:
                # sample self.kos positive items, score and grab the lowest scoring one to use
                kos_sample_item_ids = self.random_state.choice(pos_item_idxs, self.kos)
                min_item_score = 1e7
                min_item_id = -1
                for item_id in kos_sample_item_ids:
                    item_score = self._predict(user_id, item_id)
                    if item_score < min_item_score:
                        min_item_score = item_score
                        min_item_id = item_id
                if min_item_id == -1:
                    print("encountered unexpected state: didn't find min score when sampling for KOS WARP")
                    break
                pos_item_id = min_item_id
                pos_item_score = min_item_score
                # The rest is just like warp
                # sample until we find a negative item that violates the margin
                num_sampled = 0
                while num_sampled < self.max_samples:
                    num_sampled += 1
                    neg_item_id = self.random_state.randint(0, num_items)
                    if neg_item_id in pos_item_set:
                        continue
                    neg_item_score = self._predict(user_id, neg_item_id)
                    if neg_item_score <= pos_item_score - 1:
                        continue
                    # found a candidate negative item
                    loss = math.log(math.floor(num_items / num_sampled))
                    if loss > MAX_LOSS:
                        loss = MAX_LOSS

                    avg_lr = 0.0

                    # update biases
                    avg_lr += self._update_bias_grad(self.user_biases, user_id, loss, self.user_reg, \
                        item_key="user_bias", cur_iter=cur_iter)
                    avg_lr += self._update_bias_grad(self.item_biases, neg_item_id, loss, self.item_reg, \
                        item_key="item_bias", cur_iter=cur_iter)
                    avg_lr += self._update_bias_grad(self.item_biases, pos_item_id, -loss, self.item_reg, \
                        item_key="item_bias", cur_iter=cur_iter)
                    
                    user_embedding = self.user_embeddings[user_id,:]
                    pos_item_embedding = self.item_embeddings[pos_item_id,:]
                    neg_item_embedding = self.item_embeddings[neg_item_id,:]
                    # update latent factors
                    avg_lr += self._update_latent_vector_grad(self.user_embeddings, \
                                              user_id, \
                                              loss * (neg_item_embedding - pos_item_embedding) + self.user_reg * user_embedding, \
                                              self.user_reg,
                                              item_key="user_embeddings", \
                                              cur_iter=cur_iter)
                    avg_lr += self._update_latent_vector_grad(self.item_embeddings, \
                                              pos_item_id, \
                                              -loss * user_embedding + self.item_reg * pos_item_embedding, \
                                              self.item_reg,
                                              item_key="item_embeddings", \
                                              cur_iter=cur_iter)
                    avg_lr += self._update_latent_vector_grad(self.item_embeddings, \
                                              neg_item_id, \
                                              loss * user_embedding + self.item_reg * neg_item_embedding, \
                                              self.item_reg,
                                              item_key="item_embeddings", \
                                              cur_iter=cur_iter)
                    avg_lr /= (3. + (self.num_factors * 3))
                    #self.user_scale *= (1.0 + self.user_reg * avg_lr)
                    #self.item_scale *= (1.0 + self.item_reg * avg_lr)
                    break
                if self.user_scale > self.MAX_REG_SCALE or self.item_scale > self.MAX_REG_SCALE:
                    #print("regularizing!")
                    #self._regularize()
                    pass
        #self._regularize()