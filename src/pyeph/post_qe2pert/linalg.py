# Copyright (c) 2026, the PyEPH contributors.
# Migrated from https://github.com/jiangtong1000/PyEPH, revision
# 6c4693acbb69a06a5bc8b0593abde2170ff38843, under BSD-3-Clause (see LICENSE).
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
