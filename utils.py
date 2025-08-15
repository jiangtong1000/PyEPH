import numpy as np

def parse_qpoint_path(qpoint_string):
    """Parse q-point path string and generate interpolated path"""
    lines = qpoint_string.strip().split('\n')
    nqpoints = int(lines[0])
    assert len(lines) == nqpoints + 1

    high_sym_points = []
    ninterp_list = []
        
    for i in range(1, nqpoints + 1):
        parts = lines[i].split()
        qx, qy, qz = float(parts[0]), float(parts[1]), float(parts[2])
        ninterp = int(parts[3])
        high_sym_points.append([qx, qy, qz]) # (nqpoints, 3)
        ninterp_list.append(ninterp) # (nqpoints,)

    qpath = []
    qpath_labels = []
        
    for i in range(len(high_sym_points) - 1):
        q1 = np.array(high_sym_points[i])
        q2 = np.array(high_sym_points[i + 1])
        
        # qpath.append(q1)
        ninterp = ninterp_list[i] + 1
        for j in range(ninterp):
            t = j / ninterp if ninterp > 1 else 0
            q_interp = q1 + t * (q2 - q1)
            qpath.append(q_interp)

    qpath.append(q2)
    qpath = np.array(qpath)
    return qpath