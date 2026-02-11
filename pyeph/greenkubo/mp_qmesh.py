import numpy


def q_1d_mp_centered(N):
    # negative -> positive ordering, in [-0.5, 0.5)
    return (numpy.arange(N, dtype=float) + 0.5) / N - 0.5

def q_grid_mp_centered(Nx, Ny):
    qx = q_1d_mp_centered(Nx)  # shape (Nx,)
    qy = q_1d_mp_centered(Ny)  # shape (Ny,)
    QX, QY = numpy.meshgrid(qx, qy, indexing="ij")  # both (Nx,Ny)
    Q = numpy.stack([QX, QY], axis=-1)              # (Nx,Ny,2)
    return qx, qy, Q

def build_half_indices(Nx, Ny):
    # Use the Monkhorst-Pack convention
    # so no self-inverse points here when Nx and Ny are even
    assert Nx % 2 == 0 and Ny % 2 == 0, "Nx and Ny must be even"
    
    half = []
    partner = []
    for i in range(Nx):
        ip = Nx - 1 - i
        for j in range(Ny):
            jp = Ny - 1 - j

            # keep exactly one representative of each pair
            # if (i < ip) or (i == ip and j <= jp):
            if (j < jp) or (j == jp and i <= ip):
                half.append((i, j))
                partner.append((ip, jp))

    return numpy.array(half, int), numpy.array(partner, int)

def half_q_points(Nx, Ny):
    qx, qy, Q = q_grid_mp_centered(Nx, Ny)
    half_ij, partner_ij = build_half_indices(Nx, Ny)
    q_half = Q[half_ij[:,0], half_ij[:,1]]  # (nhalf,2)
    q_partner = Q[partner_ij[:,0], partner_ij[:,1]]  # (nhalf,2)
    return q_half, q_partner, half_ij, partner_ij, Q

def half_modes_to_full_modes(freq_half, q_half, q_partner, half_ij, partner_ij, Nx, Ny):
    """
    freq_half : (nmodes, nhalf_qpts) ndarray
    q_half : (nhalf, 2) ndarray
    q_partner : (nhalf, 2) ndarray
    half_ij : (nhalf, 2) ndarray
    partner_ij : (nhalf, 2) ndarray
    
    returns:
    freq_full : (nmodes, Nx, Ny) ndarray
    """
    nmodes = freq_half.shape[0]
    freq_full = numpy.zeros((nmodes, Nx, Ny))
    
    hi, hj = half_ij[:, 0], half_ij[:, 1]
    freq_full[:, hi, hj] = freq_half
    
    pi, pj = partner_ij[:, 0], partner_ij[:, 1]
    freq_full[:, pi, pj] = freq_half
    
    return freq_full

def get_mp_qmesh_info(Nx, Ny, freq_half):
    q_half, q_partner, half_ij, partner_ij, Q = half_q_points(Nx, Ny)
    freq_full = half_modes_to_full_modes(freq_half, q_half, q_partner, half_ij, partner_ij, Nx, Ny)
    return freq_full, q_half, q_partner, half_ij, partner_ij, Q