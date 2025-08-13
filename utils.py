import numpy as np

def parse_qpoint_path(qpoint_string):
    """Parse q-point path string and generate interpolated path"""
    lines = qpoint_string.strip().split('\n')
    nqpoints = int(lines[0])
    
    # Parse high symmetry points
    high_sym_points = []
    ninterp_list = []
    
    for i in range(1, nqpoints + 1):
        parts = lines[i].split()
        qx, qy, qz = float(parts[0]), float(parts[1]), float(parts[2])
        ninterp = int(parts[3])
        high_sym_points.append([qx, qy, qz])
        ninterp_list.append(ninterp)
    
    # Generate interpolated path
    qpath = []
    qpath_labels = []
    qpath_positions = [0]  # for plotting x-axis
    
    for i in range(len(high_sym_points) - 1):
        q1 = np.array(high_sym_points[i])
        q2 = np.array(high_sym_points[i + 1])
        ninterp = ninterp_list[i]
        
        # Generate interpolated points
        for j in range(ninterp):
            t = j / ninterp if ninterp > 1 else 0
            q_interp = q1 + t * (q2 - q1)
            qpath.append(q_interp)
        
        # Update positions for plotting
        if i < len(high_sym_points) - 2:  # Not the last segment
            qpath_positions.append(len(qpath))
    
    # Add the final point
    qpath.append(np.array(high_sym_points[-1]))
    qpath_positions.append(len(qpath) - 1)
    
    return np.array(qpath), qpath_positions