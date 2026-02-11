"""
Perform symmetrization and particle-hole transformation on EPH matrices.
"""
import numpy

def symmetrize_electronic_eph(g_fullgrid, rph_fullgrid, re_partialgrid, rp_partialgrid):
    """
    we construct symmetrized EPH matrices from the full grid.
    or we recover the full matrix if only given the triangular part.
    
    for i=j in [0, 1], we extract the mean
    for i=0, j=1, we directly use
    for i=1, j=0, we construct from i=0,j=1
    
    the TRS symmetry rule is g[j,i,re,rp] = g[i,j,-re,rp-re]
    so g[i,i,re,rp] = (g[i,i,re,rp]+g[i,i,-re,rp-re])/2 for i=0,1
    g[0,1,re,rp] = g[0,1,re,rp]
    g[1,0,re,rp] = g[0,1,-re,rp-re]
    g_fullgrid: (2, 2, nre, nrp, nmodes); g_fullgrid[1, 0] will not be used
    """
    nmodes = g_fullgrid.shape[-1]
    lookup_rphfull = {(int(rph_fullgrid[i, 0]), int(rph_fullgrid[i, 1])): i for i in range(rph_fullgrid.shape[0])}
    lookup_re = {(int(re_partialgrid[i, 0]), int(re_partialgrid[i, 1])): i for i in range(re_partialgrid.shape[0])}
    
    eph_partial_sym = numpy.zeros((2, 2, re_partialgrid.shape[0], rp_partialgrid.shape[0], nmodes))
    nre = re_partialgrid.shape[0]
    nrp = rp_partialgrid.shape[0]
    
    for reidx in range(nre):
        rex, rey = re_partialgrid[reidx][0], re_partialgrid[reidx][1]
        reidx_pair_in_full = lookup_re[(-rex, -rey)]
        for rpidx in range(nrp):
            rpx, rpy = rp_partialgrid[rpidx][0], rp_partialgrid[rpidx][1]
            rpidx_in_full = lookup_rphfull[(rpx, rpy)]
            rpidx_pair_in_full = lookup_rphfull[(int(rpx-rex), int(rpy-rey))]
            
            for i in range(2):
                g1 = g_fullgrid[i, i, reidx, rpidx_in_full, :]
                g2 = g_fullgrid[i, i, reidx_pair_in_full, rpidx_pair_in_full, :]
                
                # Here might need to be tested:
                # Take the mean
                # eph_partial_sym[i, i, reidx, rpidx, :] = (g1 + g2) / 2
                
                # Take the element with larger absolute value
                mask = numpy.abs(g1) >= numpy.abs(g2)
                eph_partial_sym[i, i, reidx, rpidx, :] = numpy.where(mask, g1, g2)

            eph_partial_sym[0, 1, reidx, rpidx, :] = g_fullgrid[0, 1, reidx, rpidx_in_full, :]
            eph_partial_sym[1, 0, reidx, rpidx, :] = g_fullgrid[0, 1, reidx_pair_in_full, rpidx_pair_in_full, :]
    return eph_partial_sym