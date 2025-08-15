# Phonon dispersion
## Theory and practice

Phonon dispersion, the $\lambda$-th mode $\omega_\lambda(\mathbf{q})$ is computed by solving the following eigenvalue problem:
$$
D(\mathbf{q}) |\mathbf{e}_\lambda\rangle = \omega_\lambda(\mathbf{q})^2 |\mathbf{e}_\lambda\rangle
$$
and
$$
\omega_\lambda = \begin{cases}
+\sqrt{\lambda} & \text{if } \lambda \geq 0 \\
-\sqrt{-\lambda} & \text{if } \lambda < 0 \text{ (imaginary frequency)}
\end{cases}, \quad 
\mathbf{u}_\lambda^a = \frac{|\mathbf{e}_\lambda\rangle^a}{\sqrt{M_a}}
$$
where $D(\mathbf{q})\in \mathbb{C}^{3N_{\mathrm{at}} \times 3N_{\mathrm{at}}}$ is the dynamical matrix at given q-point. 

For each atom pair $(1\le a,b\le N_{\mathrm{at}})$ and associated Cartesian directions $(\alpha,\beta\in [x,y,z])$, the dynamical matrix is constructed by Fourier transformation of the real space IFC tensor:
$$
D_{\alpha\beta}^{ab}(\mathbf{q}) = \sum_{\mathbf{R}} \frac{\Phi_{\alpha\beta}^{ab}(\mathbf{R})}{w_{\mathbf{R}}} e^{i \mathbf{q} \cdot \mathbf{R}} \cdot \frac{1}{\sqrt{M_a M_b}}
$$
where 
- $\Phi_{\alpha\beta}^{ab}(\mathbf{R})$ is the interatomic force constant (IFC) tensor which describes the force on atom $a$ in direction $\alpha$ due to displacement of atom $b$ in direction $\beta$ at lattice vector $\mathbf{R}$
- $w_{\mathbf{R}}$ is the Wigner-Seitz degeneracy weight
- $M_a, M_b$ are the atomic masses
- $\mathbf{R}$ is lattice vector connecting atom pairs.

There are infinitely many $\mathbf{R}$ vectors connecting $(a,b)$ through periodicity, 
since IFC decay rapidly with distance due to the localized nature of atomic interactions, we only need R-vectors within a certain cutoff radius to capture the essential physics. The real-space sphere radius is chosen to extend approximately to half the first Brillouin zone, ensuring adequate sampling of the reciprocal space for accurate phonon dispersion.Within this cutoff sphere, all possible lattice vectors $\mathbf{R}$ are systematically enumerated using a grid-based approach with periodic boundary conditions. For each pair of atoms $(a,b)$, we apply the Wigner-Seitz criteria to find the shortest lattice vector connecting the two atoms.


### Code Input:
- **Interatomic Force Constants (IFCs)**: $\Phi_{\alpha\beta}^{ab}(\mathbf{R})$, tensor structure with shape of $(N_{\mathrm{at}}, N_{\mathrm{at}}, N_R, 3, 3)$
  - $a, b$: atom indices in unit cell ($1 \le a,b \le N_{\mathrm{at}}$)
  - $\mathbf{R}$: lattice vector connecting unit cells ($N_R=\mathrm{q}^3$ total R-vectors)
  - $\alpha, \beta$: Cartesian directions (x, y, z)
  - Physical meaning: Force on atom $a$ in direction $\alpha$ due to displacement of atom $b$ in direction $\beta$ at lattice vector $\mathbf{R}$

## e-ph coupling matrix in real space
Instead of working everything in reciprocal space, my resulting Hamiltonian wants the electronic degree of freedom to be in real space and the phonon degree of freedom to be in reciprocal space, which means,
$$
H_{\mathrm{e}} = \sum_{i,j,R, R'} h_{ij}(R, R') a_i^\dagger(R) a_j(R')
$$
$$
H_{\mathrm{ph}} = \sum_{n,q} \omega_n(q) b_n^\dagger(q) b_n(q)
$$
$$
H_{\mathrm{e-ph}} = \sum_{i,j,R, R',q} g_{ij}^n(R, R',q) a_i^\dagger(R) a_j(R') b_n^\dagger(q) b_n(q)
$$

To get the eph coupling matrix in such form, we need to transform the data from `epr.h5`:

### Input from epr.h5:
- **ep_hop**: Electron-phonon coupling in mixed real-reciprocal space
  - **Matrix structure**: `ep_hop(Rp, Re, alpha)` for each `(iatom, iwan, jwan)` combination
  - **Python/HDF5 dimensions**: `(nrp, nre, 3)` for key `(natom, num_wann, num_wann)`
  - **Indices**:
    - `alpha`: Cartesian displacement direction (x,y,z) of atom `iatom`
    - `Re`: Electronic R-vector index (real space lattice translation)
    - `Rp`: Phononic R-vector index (real space lattice translation)
  - **Physical meaning**: Matrix element for electron hopping from unit cell 0 to `Re` when atom `iatom` is displaced in direction `α` at lattice vector `Rp`

### Transformation Algorithm:

#### Step 1: Extract ep_hop from epr.h5
For each Wannier orbital pair `(i,j)` and atom `a`:
$$
g_{ij}^{\alpha,a}(R_e, R_p) = \text{ep\_hop}[a, i, j]
$$

#### Step 2: Transform to Phonon Mode Basis
Apply phonon eigenvectors to convert from atomic displacements to phonon modes:
$$
g_{ij}^{n}(R_e, R_p, \mathbf{q}) = \sum_{a,\alpha} g_{ij}^{\alpha,a}(R_e, R_p) \cdot u_{\alpha,a}^{n}(\mathbf{q})
$$
where $u_{\alpha,a}^{n}(\mathbf{q})$ is the phonon eigenvector for mode $n$ at q-point $\mathbf{q}$.

#### Step 3: Fourier Transform Phononic Part
Convert from real-space phonon coordinates to reciprocal space:
$$
g_{ij}^{n}(R_e, \mathbf{q}) = \sum_{R_p} g_{ij}^{n}(R_e, R_p, \mathbf{q}) \cdot e^{i\mathbf{q} \cdot R_p}
$$

#### Step 4: Final Mixed Representation
The result gives the desired form:
$$
g_{ij}^n(R, R', \mathbf{q}) = g_{ij}^{n}(R_e, \mathbf{q}) \quad \text{where } R = 0, R' = R_e
$$

## Physics behind the coding
`set_cutoff_small` is the cutoff radius in real space for Wigner-Seitz cell vector search (the edge of the first Brillouin zone in reciprocal space), because the Wannier functions are localized in real space, the hopping integrals decay rapidly with the distance, we only need R-vectors within a certain cutoff radius to capture the essential physics. This is done by enumerating the 8 corners of a box in crystal coordinates, and find the maximum distance from the origin (Gamma point) to the corners.