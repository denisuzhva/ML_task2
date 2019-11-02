import numpy as np
import scipy.sparse as sp



KINO_NUM = 17770
USR_NUM = 2649429 


if __name__ == '__main__':

    sparse_dir = '../Dataset/'
    sparse_mat = sp.load_npz(sparse_dir + 'sparse.npz')
    print(sparse_mat)

    n_features = USR_NUM + KINO_NUM + KINO_NUM
    n_samples = USR_NUM * KINO_NUM
    dataset_mat = sp.dok_matrix((n_samples, n_features), dtype=np.int8)
    target_vect = np.zeros(n_samples, dtype=np.int8)

