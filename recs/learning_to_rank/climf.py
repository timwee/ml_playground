#!/usr/bin/env python

from scipy.sparse import coo_matrix, csr_matrix, diags
import numpy as np
import math
from sklearn.utils.extmath import safe_sparse_dot
from base import nnz_csrrow, LTRBase

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid_from_pred(pred):
    return pred * (1-pred)

def dsigmoid(x):
    return dsigmoid_from_pred(sigmoid(x))
    # return np.exp(-x) / (1 + np.exp(-x))**2

EPSILON = 1e-8

class CLiMF(LTRBase):
    """
    CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering
    http://baltrunas.info/papers/Shi12-climf.pdf
    """
    def __init__(self, optim, num_factors=32, reg=0.001, random_state=None, dtype=np.float32):
        self.dtype = dtype
        self.num_factors = num_factors
        self.reg = reg
        self.random_state = random_state
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        self.user_embeddings = self.item_embeddings = None
        self.optim = optim

    def _init_embeddings(self, num_users, num_items):
        self.user_embeddings = (self.random_state.rand(num_users, self.num_factors) - 0.5) / self.num_factors
        self.item_embeddings = (self.random_state.rand(num_items, self.num_factors) - 0.5) / self.num_factors
        
    def compute_objective(self, mat):
        result = -0.5 * self.reg * \
            (np.sum(self.item_embeddings * self.item_embeddings) + \
             np.sum(self.user_embeddings * self.user_embeddings))
        num_users, num_items = self.user_embeddings.shape[0], self.item_embeddings.shape[0]
        for user_id in range(num_users):
            item_ids, item_vals = nnz_csrrow(mat, user_id)
            V = self.item_embeddings[item_ids,:]
            user_embedding = self.user_embeddings[user_id]
            f = np.dot(V, user_embedding)
            result += np.sum(np.log(sigmoid(f)))
            item_ids_idxmap = {item_id : idx for idx, item_id in enumerate(item_ids)}
            for pos_item_id in item_ids:
                j = item_ids_idxmap[pos_item_id]
                fk_minus_fj = f - f[j]
                result += np.sum(np.log(1. - sigmoid(fk_minus_fj)))
        return result

    def predict_rank(self, test, train_interactions, user_features=None, item_features=None, \
                    num_threads=1, filter_train=True):
        """
        Follow signature of lightfm to use their evaluation code
        Output is a sparse matrix ranking the interactions in test parameter for each user
        """
        num_users, num_items = test.shape
        ranks = csr_matrix((np.zeros_like(test.data),
                            test.indices,
                            test.indptr), shape=test.shape)
        for usr in range(num_users):
            row_start, row_stop = test.indptr[usr], test.indptr[usr+1]
            pos_item_idxs, pos_item_vals = nnz_csrrow(test, usr)
            predictions = np.zeros((pos_item_idxs.shape[0]))
            item_ids = np.zeros((pos_item_idxs.shape[0]))
            for idx, pos_item in enumerate(pos_item_idxs):
                predictions[idx] = self._predict(usr, pos_item)
                item_ids[idx] = pos_item
            trn_item_idxs, _ = nnz_csrrow(train_interactions, usr)
            trn_item_set = set(trn_item_idxs)

            usr_item_scores = self._predict_for_user(usr)
            for item_id in range(num_items):
                if item_id in trn_item_set and filter_train:
                    continue
                cur_item_score = usr_item_scores[item_id]
                #cur_item_score = self._predict(usr, item_id)
                for i in range(row_stop - row_start):
                    if item_id != item_ids[i] and cur_item_score >= predictions[i]:
                        ranks.data[row_start + i] += 1.0
        return ranks
                
    def _predict_for_user(self, usr_id):
        """
        returns vector of predictions for all items
        """
        user_embedding = self.user_embeddings[usr_id,:]
        return np.dot(self.item_embeddings, user_embedding)

    def _predict(self, user_id, item_id):
        return np.dot(self.user_embeddings[user_id], self.item_embeddings[item_id])

    def fit(self, train, cur_iter=None, reset=True, debug=True):
        num_users, num_items = train.shape
        if self.user_embeddings is None or reset:
            print("initializing embeddings")
            self._init_embeddings(num_users, num_items)
        self._run_epoch(train, cur_iter=cur_iter, debug=debug)

    def _run_epoch(self, train, cur_iter=None, debug=True):
        if not cur_iter:
            cur_iter = 1
        num_users, num_items = train.shape
        for user_id in range(num_users):
            item_ids, item_vals = nnz_csrrow(train, user_id)
            if len(item_ids) == 0:
                # user has no items to train on
                continue
            user_embedding = self.user_embeddings[user_id]
            item_ids_idxmap = {item_id : idx for idx, item_id in enumerate(item_ids)}
            dU = -self.reg * user_embedding
            V = self.item_embeddings[item_ids,:]
            dV = np.zeros_like(V, dtype=self.dtype)
            dV -= self.reg * V
            # predictions, f_i in paper
            f = np.dot(V, user_embedding)
            assert f.shape == (len(item_ids),)

            # for first term in both gradient formulas
            g_mfi = sigmoid(-f)
            g_mfi_V = g_mfi.reshape((-1,1)) * V
            assert g_mfi_V.shape == (len(item_ids),self.num_factors)

            for pos_item_id in item_ids:
                j = item_ids_idxmap[pos_item_id]
                # shared between dU and dV
                fk_minus_fj = f - f[j]
                fj_minus_fk = -1 * fk_minus_fj

                #### compute dU
                # pos_item_idx
                dU_first_term = np.squeeze(g_mfi_V[j])

                # second term
                dU_second_term = (dsigmoid(fk_minus_fj) / (1-sigmoid(fk_minus_fj) + EPSILON))
                assert dU_second_term.shape == (len(item_ids),)
                dU_second_term = dU_second_term.reshape((-1,1)) * (V[j]-V)
                assert dU_second_term.shape == (len(item_ids), self.num_factors)
                dU_second_term = np.sum(dU_second_term, axis=0)
                assert dU_second_term.shape == dU.shape
                assert dU_first_term.shape == dU.shape
                dU += (dU_first_term + dU_second_term)

                ##### compute dV
                dV_first_term = g_mfi[j]
                dV_second_term = dsigmoid(fj_minus_fk)
                assert dV_second_term.shape == (len(item_ids),)
                dV_second_term *= ((1. / (1. - sigmoid(fj_minus_fk) + EPSILON)) - \
                                   (1. / (1. - sigmoid(fk_minus_fj) + EPSILON)))
                assert dV_second_term.shape == (len(item_ids),)
                dVj = np.sum(dV_first_term + dV_second_term)
                dV[j] += user_embedding * dVj

            # update embeddings
            self.optim(weights=self.item_embeddings, grads=-dV, idxs=item_ids, cur_iter=cur_iter, param_name="item_embeddings")
            self.optim(weights=self.user_embeddings, grads=-dU, idxs=user_id, cur_iter=cur_iter, param_name="user_embeddings")
            #self.item_embeddings[item_ids] += lr * dV
            #self.user_embeddings[user_id] += lr * dU
        if debug:
            print("Objective is: ", self.compute_objective(train))





