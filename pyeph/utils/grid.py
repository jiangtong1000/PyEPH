import numpy as np

def rgrid_2d_full(Nx, Ny):
    rx = np.arange(-Nx//2, -Nx//2 + Nx, dtype=int)  # [-Nx/2, ..., Nx/2-1] (even) or symmetric (odd)
    ry = np.arange(-Ny//2, -Ny//2 + Ny, dtype=int)
    Rx, Ry = np.meshgrid(rx, ry, indexing='ij')
    return np.column_stack([Rx.ravel(), Ry.ravel(), np.zeros(Nx*Ny, int)])


def rgrid_2d(width):
    rgrid = []
    for rx in range(-width, width + 1):
        for ry in range(-width, width + 1):
            rgrid.append([rx, ry, 0])
    return np.array(rgrid)

def generate_half_qgrids(Nx, Ny):
    qx_1d_full = (np.arange(Nx, dtype=float) + 0.5) / Nx - 0.5
    qy_1d_full = (np.arange(Ny, dtype=float) + 0.5) / Ny - 0.5
    
    qy_1d_half = qy_1d_full[qy_1d_full <= 0]
    qgrids_idx = []
    
    qx_zero_idx = np.argmin(np.abs(qx_1d_full))
    
    for iqx, qx in enumerate(qx_1d_full):
        if iqx == qx_zero_idx:  # qx ≈ 0 line
            for iqy, qy in enumerate(qy_1d_full):
                if qy <= 0:  # Only take half to avoid double counting
                    qgrids_idx.append([iqx, iqy])
        else:
            # For qx ≠ 0, only include qy <= 0 (will be paired with -qx, -qy)
            for iqy, qy in enumerate(qy_1d_half):
                qgrids_idx.append([iqx, iqy])
    
    full_qgrids_idx = []
    for iqx in range(Nx):
        for iqy in range(Ny):
            full_qgrids_idx.append([iqx, iqy])
    
    minus_qgrids_idx = []    
    for iqx, iqy in full_qgrids_idx:
        if [iqx, iqy] not in qgrids_idx:
            minus_qgrids_idx.append([iqx, iqy])
    
    minus_qgrids = np.array([[qx_1d_full[iqx], qy_1d_full[iqy], 0] for iqx, iqy in minus_qgrids_idx])
    qgrids = np.array([[qx_1d_full[iqx], qy_1d_full[iqy], 0] for iqx, iqy in qgrids_idx])
    full_qgrids = np.vstack([qgrids, minus_qgrids])
    
    partner_qgrids_idx = []
    for qx, qy, qz in minus_qgrids:
        target_qx = -qx
        target_qy = -qy
        dist = np.sqrt((target_qx - qgrids[:, 0])**2 + (target_qy - qgrids[:, 1])**2)
        argmin = np.argmin(dist)
        assert dist[argmin] < 1e-10, f"minus qgrid ({qx}, {qy}) has no partner, closest distance: {dist[argmin]}"
        partner_qgrids_idx.append(argmin)
    
    print(f"Generated MP grid: Nx={Nx}, Ny={Ny}")
    print(f"qx range: [{qx_1d_full[0]:.4f}, {qx_1d_full[-1]:.4f}], nqx={len(qx_1d_full)}")
    print(f"qy range: [{qy_1d_full[0]:.4f}, {qy_1d_full[-1]:.4f}], nqy={len(qy_1d_full)}")
    print(f"Reduced grid size: {len(qgrids)}, Full grid size: {len(full_qgrids)}")
    
    return qgrids, minus_qgrids, full_qgrids, partner_qgrids_idx