# Compare Models
1. BPR: Bayesian Personalized Ranking from Implicit Feedback https://arxiv.org/abs/1205.2618
  - Optimizes for AUC
2. (WARP) WSABIE: Scaling Up To Large Vocabulary Image Annotation http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf
  - Optimizes for Precision@k
3. (k-OS WARP) Learning to Rank Recommendations with the k-Order Statistic Loss https://research.google.com/pubs/pub41534.html
  - Optimizes for Precision@k
4. CLiMF: Learning to Maximize Reciprocal Rank with Collaborative Less-is-More Filtering http://baltrunas.info/papers/Shi12-climf.pdf
  - Optimizes for Reciprocal Rank

# Optimizers (other than vanilla SGD)
1. NAdam
2. Nesterov

# Data
- Movielens 100k https://grouplens.org/datasets/movielens/100k/
- README - http://files.grouplens.org/datasets/movielens/ml-100k-README.txt
# TODO
- speed up implementations (cython, c++, or numba/bottleneck)
