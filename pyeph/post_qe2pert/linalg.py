import numpy as np

def unpack_dyn_matrix(dyn_upper, nmodes):
    dyn_matrix = np.zeros((nmodes, nmodes), dtype=np.complex128)
    idx = 0
    for j in range(nmodes):        # Column index (0-based)
        for i in range(j + 1):   # Row index, upper triangular (0-based)
            dyn_matrix[i, j] = dyn_upper[idx]
            if i != j:
                dyn_matrix[j, i] = np.conj(dyn_upper[idx])
            idx += 1
    return dyn_matrix
