# lammps-mlip-package
A user package of LAMMPS software enabling simulations using linearized machine learning potentials

Machine learning potentials for a wide range of systems can be found in the website. If you use **lammps-mlip** package and machine learning potentials in the repository for academic purposes, please cite the following article [1].

[1] A. Seko, A. Togo and I. Tanaka, "Group-theoretical high-order rotational invariants for structural representations: Application to linearized machine learning interatomic potential", Phys. Rev. B 99, 214108 (2019).

## Installation
```shell
git clone git@github.com:quantum-mtl/lammps-mlip-package.git
cd lammps-mlip-package
git submodule update --init --recursive --recommend-shallow --depth 1 
```

We provide two options for building this package, (1) buidling locally (2) using Docker.

### (1) Building lammps with lammps-mlip package

1. Copy all the components in the **lammps-mlip** package to the latest lammps source code directory as
```shell
cp -r lammps-mlip/lib/mlip $(lammps_src)/lib
cp -r lammps-mlip/src/USER-MLIP $(lammps_src)/src
```
2. Modify `$(lammps_src)/lib/mlip/Makefile.lammps` to specify an installed directory of the boost library.

3. Add "user-mlip" to variable PACKUSER defined in $(lammps_src)/src/Makefile and activate user-mlip package as
```shell
vi $(lammps_src)/src/Makefile
    PACKUSER = user-atc user-awpmd user-cgdna user-cgsdk user-colvars \
        user-diffraction user-dpd user-drude user-eff user-fep user-h5md \
        user-intel user-lb user-manifold user-meamc user-mgpt user-misc user-molfile \
        user-netcdf user-omp user-phonon user-qmmm user-qtb \
        user-quip user-reaxc user-smd user-smtbq user-sph user-tally \
        user-vtk user-mlip

make yes-user-mlip
```
4. Build lammps binary files
```shell
make serial -j 36
```

### (2) Run lammps-mlip in docker
1. build and run a docker container
```shell
docker build -t lammps -f docker/Dockerfile .
docker run -it -v $(PWD):/workspace -t lammps
```

in `lammps` container
```
cd /workspace
./docker/install.sh
```
Now `lmp_serial` is built under `/workspace`.


## Lammps input commands to specify a machine learning potential

The following lammps input commands specify a machine learning potential.
```
pair_style  mlip_pair
pair_coeff * * pyml.lammps.mlip Ti Al    
```
or
```
pair_style  mlip_gtinv
pair_coeff * * pyml.lammps.mlip Ti Al    
```

## Examples
```
cd example
./../lmp_serial -in in.mlip
```

## Development

### Unit test
Unit tests are managed by GoogleTest.
If you add a unit test, modify `test/CMakeLists.txt` and `
```
cd test
mkdir build && cd build
cmake ..
make -j 32
ctest -vv
```

### Regression test
- structure: FCC-perturbed TiAl3 (32 atoms)
- potential: gtinv-p=2-l=4-order=2
- 100 timesteps with NVE

```
cd test/regression
./test.sh
```

### performance
- perf
```
sudo perf record --call-graph lbr ./../../lmp_serial -in in.mlip
sudo perf report
```

- Flamegraph
```
cd Flamegraph
sudo cp ../perf.data .
sudo perf script -f | ./stackcollapse-perf.pl | ./flamegraph.pl >perf.svg
```
