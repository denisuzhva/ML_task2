import numpy as np
import scipy.sparse as sp



KINO_NUM = 17770
USR_NUM = 2649429 


if __name__ == '__main__':

    sparse_dir = '../../Dataset/netflix/'
    sparse_mat = sp.load_npz(sparse_dir + 'netfilx_sparse_mat.npz')
    sp_rows, sp_cols = sparse_mat.nonzero() # rows = users, cols = kinos
    true_len = sp_rows.shape[0]
    print('Entries: %i' % true_len)
    dens = true_len / (KINO_NUM * USR_NUM)
    print('Density: %.8f' % dens)

    n_features = USR_NUM + KINO_NUM
    n_samples = true_len
    dataset_mat = sp.dok_matrix((n_samples, n_features), dtype=np.int8)
    target_vect = np.zeros(n_samples, dtype=np.int8)

    # Process data
    ittt = 0
    for sp_row, sp_col in zip(sp_rows, sp_cols):
        if ittt % 100000 == 0:
            print('Element #%i' % ittt)
        dataset_mat[ittt, sp_row] = 1
        dataset_mat[ittt, USR_NUM + sp_col] = 1
        #dataset_mat[ittt, USR_NUM + KINO_NUM : USR_NUM + KINO_NUM + KINO_NUM] = sparse_mat[sp_row, :]

        ittt += 1
        if ittt == 1:
            pass

    dataset_mat_csr = dataset_mat.tocsr()
    sp.save_npz('../Dataset/dataset.npz', dataset_mat_csr)