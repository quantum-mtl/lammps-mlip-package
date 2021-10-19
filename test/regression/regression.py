import unittest
import os
import subprocess
import re
import sys
import argparse


L1_TOL = 1e-4  # tolerance for L1 displacements of two configurations


class RegressionTest(unittest.TestCase):

    def setUp(self):
        self.lammps_path = os.path.join(os.path.dirname(__file__), '..', '..', 'lmp_mlip_kokkos')
        potential_list = [
            'pair-60',
            'gtinv-197', 'gtinv-197',  # model_type=2
            'gtinv-820',  # model_type=3
            'gtinv-434',  # model_type=4
        ]
        self.input_path = [p + '.in' for p in potential_list]
        self.dump_path = ['dump.atom.' + p for p in potential_list]
        self.log_path = ['log.lammps.' + p for p in potential_list]
        if args.arch == 'OpenMP':
            self.lmp_option = [
                [],
                ['-sf', 'kk', '-k', 'on', 't', '2', '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on'],
                ['-sf', 'kk', '-k', 'on', 't', '2', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on'],
                ['-sf', 'kk', '-k', 'on', 't', '2', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on'],
                ['-sf', 'kk', '-k', 'on', 't', '2', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on'],
            ]
        elif args.arch == 'Cuda':
            self.lmp_option = [
                [],
                ['-sf', 'kk', '-k', 'on', 'g', '1', '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on'],
                ['-sf', 'kk', '-k', 'on', 'g', '1', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on'],
                ['-sf', 'kk', '-k', 'on', 'g', '1', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on'],
                ['-sf', 'kk', '-k', 'on', 'g', '1', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on'],
            ]


    def test_regression(self):
        for input_, dump, log, opt in zip(self.input_path, self.dump_path, self.log_path, self.lmp_option):
            with self.subTest(input_=input_, dump=dump, log=log, opt=opt):
                subprocess.run([self.lammps_path, '-in', input_, *opt])

                # Compare final coordinates of atoms
                with open('dump.atom', 'r') as f:
                    coords_actual = get_atom_coords(f.read().splitlines())
                with open(dump, 'r') as f:
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
                with open(log, 'r') as f:
                    final_temperature_expect, final_total_energy_expect, final_pressure_expect = get_thermo_info(f.read().splitlines())

                self.assertAlmostEqual(final_temperature_actual, final_temperature_expect)
                self.assertAlmostEqual(final_total_energy_actual, final_total_energy_expect)
                self.assertAlmostEqual(final_pressure_acutal, final_pressure_expect)


    def test_triclinic_run0(self):
        # use gtinv-197 with full-neighbor style and newton on
        input_ = 'gtinv-197-tricli.in'
        log = 'log.lammps.gtinv-197.conv64'
        opt = self.lmp_option[2]
        subprocess.run([self.lammps_path, '-in', input_, *opt])

        # Compare thermo info
        with open('log.lammps', 'r') as f:
            initial_temperature_actual, initial_total_energy_actual, initial_pressure_actual = get_thermo_info(f.read().splitlines())
        with open(log, 'r') as f:
            initial_temperature_expect, initial_total_energy_expect, initial_pressure_expect = get_thermo_info(f.read().splitlines())

        self.assertAlmostEqual(initial_temperature_actual, initial_temperature_expect)
        self.assertAlmostEqual(initial_total_energy_actual, initial_total_energy_expect)
        self.assertAlmostEqual(initial_pressure_actual, initial_pressure_expect, delta=0.1)


    def test_stress(self):
        # use seko-sensei's stress tensor obtained from ev_tally as a reference
        input_ = 'gtinv-197-stress.in'
        log = 'log.lammps.gtinv-197.stress.ev_tally'
        opt = self.lmp_option[2]
        subprocess.run([self.lammps_path, '-in', input_, *opt])

        # Compare stress info
        with open('log.lammps', 'r') as f:
            pxx_actual, pyy_actual, pzz_actual, pxy_actual, pxz_actual, pyz_actual = get_stress_info(f.read().splitlines())
        with open(log, 'r') as f:
            pxx_expect, pyy_expect, pzz_expect, pxy_expect, pxz_expect, pyz_expect = get_stress_info(f.read().splitlines())
        self.assertAlmostEqual(pxx_actual, pxx_expect)
        self.assertAlmostEqual(pyy_actual, pyy_expect)
        self.assertAlmostEqual(pzz_actual, pzz_expect)
        self.assertAlmostEqual(pxy_actual, pxy_expect)
        self.assertAlmostEqual(pxz_actual, pxz_expect)
        self.assertAlmostEqual(pyz_actual, pyz_expect)


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


def get_stress_info(lines):
    """
    Parse 'Step Temp E_pair E_mol TotEng Press Volume Pxx Pyy Pzz Pxy Pxz Pyz'-lines in log.lammps
    """
    idx_after_thermo_info = None
    for i, line in enumerate(lines):
        if line.startswith("Loop time of"):
            idx_after_thermo_info = i
            break
    assert idx_after_thermo_info >= 1

    thermo_info_line = lines[idx_after_thermo_info - 1]
    step, temperature, pair_energy, mol_energy, total_energy, pressure, volume, pxx, pyy, pzz, pxy, pxz, pyz = tuple(map(float, re.split(r'\s+', thermo_info_line)[1:14]))
    return pxx, pyy, pzz, pxy, pxz, pyz


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', default='OpenMP', help='Specify Kokkos architecture from OpenMP or Cuda')
    parser.add_argument('unittest_args', nargs='*')
    args = parser.parse_args()
    sys.argv[1:] = args.unittest_args
    unittest.main()
