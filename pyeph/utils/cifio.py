from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

def cif_to_qe_blocks(
    cif_path: str,
    output_path: str = "geometry.in",
    primitive: bool = True,
    symprec: float = 1e-3,
    angle_tolerance: float = 5.0,
    positions_unit: str = "angstrom",   # "angstrom" or "crystal"
    sort_sites: bool = False,
):
    parser = CifParser(cif_path, occupancy_tolerance=1.05)
    s = parser.parse_structures(primitive=False)[0]

    sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance)

    if primitive:
        s_std = sga.get_primitive_standard_structure()
        cell_type = "Primitive standard cell"
    else:
        s_std = sga.get_conventional_standard_structure()
        cell_type = "Conventional standard cell"

    # QE needs a fully ordered model
    if not s_std.is_ordered:
        raise ValueError(
            "Structure is NOT ordered (partial occupancy/disorder). "
            "Choose an ordered model before QE."
        )

    if sort_sites:
        s_std = s_std.get_sorted_structure()

    a, b, c = s_std.lattice.abc
    alpha, beta, gamma = s_std.lattice.angles

    # Detect species order as they appear in the final structure
    species_order = []
    seen = set()
    for site in s_std:
        # Use .symbol to avoid "Fe2+" issues unless intended
        el = site.specie.symbol 
        if el not in seen:
            species_order.append(el)
            seen.add(el)

    with open(output_path, "w") as fout:
        # Pymatgen comment syntax changed to '!' for QE compatibility
        fout.write("! === CIF Information ===\n")
        fout.write(f"! Formula: {s.composition.reduced_formula}\n")
        fout.write(f"! Ordered: {s.is_ordered}\n")
        fout.write(f"! Lattice (Angstrom):\n! {s.lattice.matrix}\n")

        fout.write("\n! === Symmetry Information ===\n")
        fout.write(f"! Space group: {sga.get_space_group_symbol()} ({sga.get_space_group_number()})\n")
        fout.write(f"! Transformation: {cell_type}\n")
        fout.write(f"! abc (A): {a:.8f} {b:.8f} {c:.8f}\n")
        fout.write(f"! angles (deg): {alpha:.6f} {beta:.6f} {gamma:.6f}\n")

        fout.write("\n! === QE Geometry Block ===\n")
        fout.write("CELL_PARAMETERS (angstrom)\n")
        # Increased to .12f for ibrav=0 safety
        for v in s_std.lattice.matrix:
            fout.write(f"{v[0]:18.12f} {v[1]:18.12f} {v[2]:18.12f}\n")

        if positions_unit.lower() in ["angstrom", "cart", "cartesian"]:
            fout.write("\nATOMIC_POSITIONS (angstrom)\n")
            for site in s_std:
                x, y, z = site.coords
                # Using 18.12f formatting for alignment and precision
                fout.write(f"{site.specie.symbol:4s} {x:18.12f} {y:18.12f} {z:18.12f}\n")
        elif positions_unit.lower() in ["crystal", "frac", "fractional"]:
            fout.write("\nATOMIC_POSITIONS (crystal)\n")
            for site in s_std:
                x, y, z = site.frac_coords
                fout.write(f"{site.specie.symbol:4s} {x:18.12f} {y:18.12f} {z:18.12f}\n")
        else:
            raise ValueError("positions_unit must be 'angstrom' or 'crystal'.")

        fout.write("\n! # Suggested QE SYSTEM fields\n")
        fout.write(f"! nat  = {len(s_std)}\n")
        fout.write(f"! ntyp = {len(species_order)}\n")
        fout.write(f"! species order = {species_order}\n")

    print(f"Done. Wrote geometry to {output_path}")