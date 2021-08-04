import numpy as np
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData


if __name__ == '__main__':
    # TiAl3
    matrix = 4.0 * np.eye(3)
    species = ['Ti', 'Al', 'Al', 'Al']
    frac_coords = [
        [0, 0, 0],
        [0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0],
    ]
    structure = Structure(matrix, species, frac_coords)
    structure.make_supercell([2, 2, 2])
    structure.perturb(0.1)

    ld = LammpsData.from_structure(structure, atom_style="atomic")
    # output file need to be fixed
    ld.write_file("structure.lammps")
