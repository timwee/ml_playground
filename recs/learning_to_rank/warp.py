from scipy.sparse import coo_matrix, csr_matrix, diags
import numpy as np
import math
from sklearn.utils.extmath import safe_sparse_dot

MAX_REG_SCALE = 1000000
MAX_LOSS = 100

def nnz_csrrow(m, rownum):
    """ Returns col indices and data that are nonzero in csr matrix m given rownum """
    start, stop = m.indptr[rownum], m.indptr[rownum + 1]
    return m.indices[start:stop], m.data[start:stop]

class WARPNaive:

    def __init__(self, num_factors=32, max_samples=125, lr=1e-7, item_reg=0.0, user_reg=0.0, \
                random_state=None, dtype=np.float32):
        """
        Parameters
        ----------
        num_factors : int
                      number of latent factors for users and items
        max_samples : int
                      max number of samples for getting a negative sample violating the margin
        lr : float
             learning rate
        item_reg : float
                   regularization for item factors
        user_reg : float
                   regularization for user factors
        """
        self.num_factors = num_factors
        self.max_samples = max_samples
        self.lr = lr
        self.item_reg = item_reg
        self.user_reg = user_reg
        self.dtype=dtype
        self.random_state = random_state
        self.item_scale = 1.0
        self.user_scale = 1.0
        self.item_embeddings = self.user_embeddings = self.user_biases = self.item_biases = None
        if self.random_state is None:
            self.random_state = np.random.RandomState()
            
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
            for item_id in range(num_items):
                if item_id in trn_item_set and filter_train:
                    continue
                
                cur_item_score = self._predict(usr, item_id)
                for i in range(row_stop - row_start):
                    if item_id != item_ids[i] and cur_item_score >= predictions[i]:
                        ranks.data[row_start + i] += 1.0
        return ranks
                
            
        
    def _predict(self, usr_id, item_id):
        user_bias, user_embedding = self.user_biases[usr_id], self.user_embeddings[usr_id,:]
        item_bias, item_embedding = self.item_biases[item_id], self.item_embeddings[item_id,:]
        return safe_sparse_dot(user_embedding, item_embedding) + user_bias + item_bias

    def _initialize_embeddings(self, num_users, num_items):
        stddev = 1. / math.sqrt(self.num_factors)
        self.item_embeddings = (self.random_state.rand(num_items, self.num_factors) - 0.5) / self.num_factors
        self.user_embeddings = (self.random_state.rand(num_users, self.num_factors) - 0.5) / self.num_factors
       
    def _initialize_biases(self, num_users, num_items):
        self.user_biases = np.zeros((num_users,), dtype=self.dtype)
        self.item_biases = np.zeros((num_items,), dtype=self.dtype)

    def _update_bias_grad(self, bias_vector, bias_idx, grad, reg):
        bias_vector[bias_idx] -= self.lr * grad
        # account for regularization
        bias_vector[bias_idx] *= (1.0 + reg * self.lr)
        
    def _update_latent_vector_grad(self, latent_vector, idx, grad, reg):
        latent_vector[idx,:] -= self.lr * grad
        # regularization
        latent_vector[idx,:] *= (1.0 + reg * self.lr)
        
    def _regularize(self):
        self.item_embeddings /= self.item_scale
        self.item_biases /= self.item_scale
        self.user_embeddings /= self.user_scale
        self.user_biases /= self.user_scale
        self.item_scale = 1.0
        self.user_scale = 1.0
    
    def fit(self, train, epochs=1, reset=True):
        num_users, num_items = train.shape
        if self.item_embeddings is None or reset:
            print("initializing embeddings and biases")
            self._initialize_embeddings(num_users, num_items)
            self._initialize_biases(num_users, num_items)
        for epoch in range(epochs):
            self._train_epoch(train)
        
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
                if self.user_scale > MAX_REG_SCALE or self.item_scale > MAX_REG_SCALE:
                    #print("regularizing!")
                    self._regularize()
        self._regularize()