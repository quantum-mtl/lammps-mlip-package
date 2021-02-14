import unittest
import os
import subprocess
import re


L1_TOL = 1e-4  # tolerance for L1 displacements of two configurations


class RegressionTest(unittest.TestCase):

    def test(self):
        lammps_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lmp_serial')
        input_path = 'in.mlip'

        output_path = 'dump.atom'
        expect_path = 'dump.atom.regression'

        subprocess.run([lammps_path, '-in', input_path])

        with open(output_path, 'r') as f:
            coords_actual = get_atom_coords(f.read().splitlines())
        with open(expect_path, 'r') as f:
            coords_expect = get_atom_coords(f.read().splitlines())

        self.assertEqual(len(coords_actual), len(coords_expect))
        l1 = 0.0
        for actual, expect in zip(coords_actual, coords_expect):
            l1 += sum([abs(actual[x] - expect[x]) for x in range(3)])
        l1 /= len(coords_actual)
        self.assertTrue(l1 < L1_TOL)


def get_atom_coords(lines):
    coords = []
    for line in lines[9:]:
        xyz = list(map(float, re.split(r'\s+', line)[2:5]))
        coords.append(xyz)
    return coords


if __name__ == '__main__':
    unittest.main()
