from scipy.constants import physical_constants as c
import numpy
import scipy.sparse


def wannier_center_and_cell_vecs_for_simple_2D(a: float, b: float):
    """
    Get the wannier center and cell vectors for a simple 2D lattice.
    a: lattice constant along x
    b: lattice constant along y
    Returns:
        wannier_center: the wannier center
        cell_vecs: the cell vectors
    """
    cell_vecs = numpy.array([[a, 0], [0, b]])
    wannier_center_pos = numpy.array([
        [0.25 * a, 0.25 * b],
        [0.75 * a, 0.75 * b]
    ])
    return wannier_center_pos, cell_vecs


au2ev = c["Hartree energy in eV"][0]
K2au = c["kelvin-hartree relationship"][0]
cm2au = 1.0e2 * c["inverse meter-hertz relationship"][0] / c["hartree-hertz relationship"][0]
fs2au = 1.0e-15 / c["atomic unit of time"][0]
ryd_to_ev = 13.605698066
ryd_to_mev = ryd_to_ev * 1000.0

mobility2au = au2ev * c["atomic unit of time"][0] / (c["atomic unit of length"][0] * 100) ** 2


def get_deltaV_from_g(gP, wP, temperature):
    # all in unit of V
    deltaV = gP * wP / numpy.sqrt(numpy.tanh(wP / (2 * temperature)))
    return deltaV


def get_g_from_deltaV(deltaV, wP, temperature):
    # all in unit of V
    gP = deltaV * numpy.sqrt(numpy.tanh(wP / (2 * temperature))) / wP
    return gP

# for debugging (my old code uses different ordering of sites)
def get_map(nx, ny):
    map = []
    total_sites = nx * ny * 2
    for i in range(0, total_sites, nx * 2):
        original_idx = numpy.arange(i, i + nx * 2)
        original_idx = original_idx.reshape(nx, 2).T
        original_idx = original_idx.flatten()
        map.extend(original_idx)
    return map


def _h5_format(obj):
    fmt = obj.attrs.get("format", None)
    if isinstance(fmt, bytes):
        fmt = fmt.decode()
    return fmt


def write_hdf5_csr_matrix(parent, name, matrix, compression="gzip", compression_opts=4):
    matrix = matrix.tocsr()
    group = parent.create_group(name)
    group.attrs["format"] = "csr"
    group.attrs["shape"] = matrix.shape
    group.create_dataset(
        "data",
        data=matrix.data,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )
    group.create_dataset(
        "indices",
        data=matrix.indices,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )
    group.create_dataset(
        "indptr",
        data=matrix.indptr,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )


def write_hdf5_csr_matrix_list(parent, name, matrices, compression="gzip", compression_opts=4):
    matrices = [matrix.tocsr() for matrix in matrices]
    nmat = len(matrices)
    nrow, ncol = matrices[0].shape
    nnz_offsets = numpy.empty(nmat + 1, dtype=numpy.int64)
    nnz_offsets[0] = 0
    for imat, matrix in enumerate(matrices):
        if matrix.shape != (nrow, ncol):
            raise ValueError(f"All matrices in {name} must have the same shape")
        nnz_offsets[imat + 1] = nnz_offsets[imat] + matrix.nnz

    group = parent.create_group(name)
    group.attrs["format"] = "csr_list"
    group.attrs["shape"] = (nmat, nrow, ncol)
    group.create_dataset(
        "nnz_offsets",
        data=nnz_offsets,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )
    data = group.create_dataset(
        "data",
        shape=(int(nnz_offsets[-1]),),
        dtype=matrices[0].data.dtype,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )
    indices = group.create_dataset(
        "indices",
        shape=(int(nnz_offsets[-1]),),
        dtype=matrices[0].indices.dtype,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )
    indptr = group.create_dataset(
        "indptr",
        shape=(nmat, nrow + 1),
        dtype=matrices[0].indptr.dtype,
        compression=compression,
        compression_opts=compression_opts,
        shuffle=True,
    )

    for imat, matrix in enumerate(matrices):
        start = nnz_offsets[imat]
        stop = nnz_offsets[imat + 1]
        data[start:stop] = matrix.data
        indices[start:stop] = matrix.indices
        indptr[imat] = matrix.indptr


def load_hdf5_matrix(obj, dense=False):
    if _h5_format(obj) == "csr":
        shape = tuple(numpy.asarray(obj.attrs["shape"], dtype=numpy.int64))
        matrix = scipy.sparse.csr_matrix(
            (obj["data"][:], obj["indices"][:], obj["indptr"][:]),
            shape=shape,
        )
        return matrix.toarray() if dense else matrix
    matrix = obj[()]
    return matrix


def get_hdf5_matrix_list_shape(obj):
    if _h5_format(obj) == "csr_list":
        return tuple(numpy.asarray(obj.attrs["shape"], dtype=numpy.int64))
    return obj.shape


def load_hdf5_matrix_list_item(obj, index, dense=False):
    if _h5_format(obj) == "csr_list":
        shape = tuple(numpy.asarray(obj.attrs["shape"], dtype=numpy.int64)[1:])
        start = int(obj["nnz_offsets"][index])
        stop = int(obj["nnz_offsets"][index + 1])
        matrix = scipy.sparse.csr_matrix(
            (
                obj["data"][start:stop],
                obj["indices"][start:stop],
                obj["indptr"][index],
            ),
            shape=shape,
        )
        return matrix.toarray() if dense else matrix
    matrix = obj[index]
    return matrix
