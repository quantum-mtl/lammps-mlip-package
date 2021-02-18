import unittest
import os
import subprocess
import re


L1_TOL = 1e-4  # tolerance for L1 displacements of two configurations


class RegressionTest(unittest.TestCase):

    def setUp(self):
        lammps_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lmp_serial')
        input_path = 'in.mlip'

        subprocess.run([lammps_path, '-in', input_path])

    def test_regression(self):
        # Compare final coordinates of atoms
        with open('dump.atom', 'r') as f:
            coords_actual = get_atom_coords(f.read().splitlines())
        with open('dump.atom.regression', 'r') as f:
            coords_expect = get_atom_coords(f.read().splitlines())

        self.assertEqual(len(coords_actual), len(coords_expect))
        l1 = 0.0
        for actual, expect in zip(coords_actual, coords_expect):
            l1 += sum([abs(actual[x] - expect[x]) for x in range(3)])
        l1 /= 3 * len(coords_actual)
        self.assertTrue(l1 < L1_TOL, "Mean coords displacement = {}".format(l1))

        # Compare thermo info
        with open('log.lammps', 'r') as f:
            final_temperature_actual, final_total_energy_actual, final_pressure_acutal = get_thermo_info(f.read().splitlines())
        with open('log.lammps.regression', 'r') as f:
            final_temperature_expect, final_total_energy_expect, final_pressure_expect = get_thermo_info(f.read().splitlines())

        self.assertAlmostEqual(final_temperature_actual, final_temperature_expect)
        self.assertAlmostEqual(final_total_energy_actual, final_total_energy_expect)
        self.assertAlmostEqual(final_pressure_acutal, final_pressure_expect)


def get_atom_coords(lines):
    coords = []
    for line in lines[9:]:
        xyz = list(map(float, re.split(r'\s+', line)[2:5]))
        coords.append(xyz)
    return coords


def get_thermo_info(lines):
    """
    Parse 'Step Temp E_pair E_mol TotEng Press'-lines in log.lammps
    """
    idx_after_thermo_info = None
    for i, line in enumerate(lines):
        if line.startswith("Loop time of"):
            idx_after_thermo_info = i
            break
    assert idx_after_thermo_info >= 1

    thermo_info_line = lines[idx_after_thermo_info - 1]
    step, temperature, pair_energy, mol_energy, total_energy, pressure = tuple(map(float, re.split(r'\s+', thermo_info_line)[1:7]))
    return temperature, total_energy, pressure


if __name__ == '__main__':
    unittest.main()
