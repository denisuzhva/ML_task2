import numpy as np
import scipy.sparse as sp



a = sp.dok_matrix((2, 2), dtype=np.float)
a[0, 1] = 1.0
a[1, 0] = 1.0
a_csr = a.tocsr()
a_dense = a_csr.todense()

b = a
b[0, 0] = 0.5
b[1, 1] = 0.5
b_csr = b.tocsr()
b_dense = np.array(b_csr.todense())


v = np.array([2, 3]).reshape(-1, 1)
print(b_dense)
print(v)
print(np.multiply(v, b_dense))



