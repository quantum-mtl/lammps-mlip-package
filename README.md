# lammps-mlip-package
A user package of LAMMPS software enabling simulations using linearized machine learning potentials

Machine learning potentials for a wide range of systems can be found in the website. If you use **lammps-mlip** package and machine learning potentials in the repository for academic purposes, please cite the following article [1].

[1] A. Seko, A. Togo and I. Tanaka, "Group-theoretical high-order rotational invariants for structural representations: Application to linearized machine learning interatomic potential", Phys. Rev. B 99, 214108 (2019).

## Installation
First, clone this repository:
```shell
git clone git@github.com:quantum-mtl/lammps-mlip-package.git
cd lammps-mlip-package
git submodule update --init --recursive --recommend-shallow --depth 1 
```

We provide three options for building this package:
(1) using singularity (2) using Docker, and (3) buidling locally.

Summary of environments provided by this repository:
|             | OpenMP w/o OpenMPI | OpenMP w/ OpenMPI | Cuda w/o OpenMPI    | Cuda w/ OpenMPI     |
|-------------|--------------------|-------------------|---------------------|---------------------|
| Docker      | `Dockerfile`       | ✖️                 | `Dockerfile.gpu`    | ✖️                   |
| Singularity | WIP                | WIP               | `ubuntu_nvidia.def` | `ubuntu_nvidia.def` |

### (1-1) Singularity with GPU
1. Make sure you install [singularity](https://sylabs.io/guides/3.0/user-guide/installation.html)
2. Make sure your machine has a GPU and you install nvidia-driver.

Create a container with cuda-aware OpenMPI:
```shell
singularity build --fakeroot mlip.sif containers/ubuntu_nvidia.def
singularity run --nv mlip.sif
```

3. Build LAMMPS in `mlip` container

#### With CMake

Please change `Kokkos_ARCH_SNB` and `Kokkos_ARCH_PASCAL60` based on [architecture](https://docs.lammps.org/Build_extras.html#available-architecture-settings).
```shell
cp -r src/USER-MLIP lammps/src
cp containers/CMakeLists.txt lammps/cmake/CMakeLists.txt
cp containers/USER-MLIP.cmake lammps/cmake/Modules/Packages/.
cd lammps
mkdir build_kokkos_cuda && cd build_kokkos_cuda
cmake -D LAMMPS_MACHINE=cuda \
      -D Kokkos_ARCH_SNB=yes \
      -D Kokkos_ARCH_PASCAL60=yes \
      -D Kokkos_ENABLE_CUDA=yes \
      -D Kokkos_ENABLE_OPENMP=yes \
      -D Kokkos_ENABLE_SERIAL=yes \
      -D BUILD_OMP=yes \
      -D CMAKE_CXX_FLAGS=-std=c++17 \
      -D CMAKE_CXX_COMPILER=$(realpath $(pwd)/../lib/kokkos/bin/nvcc_wrapper) \
      -D PKG_KOKKOS=yes \
      -D PKG_MANYBODY=yes \
      -D PKG_USER-MLIP=yes \
      ../cmake/
make -j 8
```
Now `lmp_cuda` is built under `lammps/build_kokkos_cuda`.

#### With make
Replace `8` with the appropriate number of cores:
```
KOKKOS_DEVICES=Cuda ./containers/install.sh 8
```
You may need to change `Makefile.mlip_kokkos` for Kokkos options

### (1-2) Singularity without GPU
```shell
cp -r src/USER-MLIP lammps/src
cp containers/CMakeLists.txt lammps/cmake/CMakeLists.txt
cp containers/USER-MLIP.cmake lammps/cmake/Modules/Packages/.
cd lammps
mkdir build_kokkos_openmp && cd build_kokkos_openmp
cmake -D LAMMPS_MACHINE=openmp \
      -D Kokkos_ARCH_SNB=yes \
      -D Kokkos_ENABLE_OPENMP=yes \
      -D Kokkos_ENABLE_SERIAL=yes \
      -D BUILD_OMP=yes \
      -D CMAKE_CXX_FLAGS=-std=c++17 \
      -D PKG_KOKKOS=yes \
      -D PKG_MANYBODY=yes \
      -D PKG_USER-MLIP=yes \
      ../cmake/
```
Now `lmp_openmp` is built under `lammps/build_kokkos_openmp`.

### (2-1) Docker with GPU
1. Make sure your machine has a GPU and you install nvidia-driver.
2. Make sure you install docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
```shell
docker build -t lammps-gpu -f containers/Dockerfile.gpu .
docker run -it -v $(pwd):/workspace -t lammps-gpu
```

3. Build LAMMPS in `mlip` container
Replace `8` with the appropriate number of cores:
```
cd /workspace
KOKKOS_DEVICES=Cuda ./containers/install.sh 8
```
Now `lmp_mlip_kokkos` is built under `/workspace`.

### (2-2) Docker without GPU
1. Make sure you install docker.
```shell
docker build -t lammps -f containers/Dockerfile .
docker run -it -v $(pwd):/workspace -t lammps
```

2. Build LAMMPS in `mlip` container

Replace `8` with the appropriate number of cores
```shell
cd /workspace
./containers/install.sh 8
```
Now `lmp_mlip_kokkos` is built under `/workspace`.

### (3) Local without Kokkos

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
```shell
cd example
export LAMMPS_BIN=$(realpath ../lammps/build/lmp_cuda)
```

Without Kokkos:
```shell
./../lmp_mlip_kokkos -in in.mlip
```

1 node, 2 MPI tasks/node, 2 GPUs/node
```shell
mpirun -np 2 ${LAMMPS_BIN} -in in.mlip -kokkos on gpus 2 -suffix kk -pk kokkos neigh full newton on cuda/aware on
```

1 node, 1 MPI tasks/node, 2 GPUs/node
```shell
mpirun -np 1 ${LAMMPS_BIN} -in in.mlip -kokkos on gpus 2 -suffix kk -pk kokkos neigh full newton on cuda/aware on
```

1 node, 2 MPI tasks/node, 8 OpenMP threads/task
```shell
mpirun -np 2 ../lammps/build/lmp_openmp -in in.mlip -kokkos on t 8 -suffix kk -pk kokkos neigh full newton on
```

## Python interface

### Installation with CMake

```shell
cp -r src/USER-MLIP lammps/src
cp containers/CMakeLists.txt lammps/cmake/CMakeLists.txt
cp containers/USER-MLIP.cmake lammps/cmake/Modules/Packages/.
cd lammps
mkdir build_kokkos_cuda && cd build_kokkos_cuda
cmake -D LAMMPS_MACHINE=cuda \
      -D LAMMPS_EXCEPTIONS=yes \
      -D Kokkos_ARCH_SNB=yes \
      -D Kokkos_ARCH_PASCAL60=yes \
      -D Kokkos_ENABLE_CUDA=yes \
      -D Kokkos_ENABLE_OPENMP=yes \
      -D Kokkos_ENABLE_SERIAL=yes \
      -D BUILD_OMP=yes \
      -D BUILD_SHARED_LIBS=yes \
      -D CMAKE_CXX_FLAGS=-std=c++17 \
      -D CMAKE_CXX_COMPILER=$(realpath $(pwd)/../lib/kokkos/bin/nvcc_wrapper) \
      -D PKG_PYTHON=yes \
      -D PKG_KOKKOS=yes \
      -D PKG_MANYBODY=yes \
      -D PKG_USER-MLIP=yes \
      ../cmake/
make -j 8
make install
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH
```
Now `lmp_cuda` and `liblammps_cuda.so ` are built under `lammps/build_kokkos_cuda`.

### Simple usages

```shell
cd exmaples/
python3
```

```python
from lammps import PyLammps
# load `liblammps_{name}.so`
cmdargs = ['-log', 'none', '-kokkos', 'on', 'gpus', '1', '-suffix', 'kk', '-pk', 'kokkos', 'neigh', 'half', 'newton', 'on']
l = PyLammps(name='cuda', cmdargs=cmdargs)

# run simulation
l.command('atom_modify map yes')  # required to extract per-atom info
l.file('in.mlip')

# retrieve system property
# See https://docs.lammps.org/Python_properties.html
print("Potential energy: ", l.eval("pe"))

# retrieve per-atom property
# See https://docs.lammps.org/Python_atoms.html
for i in range(l.system.natoms):
    print(f"position[{i}] = {l.atoms[i].position}")

l.close()
```

## Development

### Installation
Run docker container
```
make init-docker
make create-container
```

Build LAMMPS (after attaching a docker container)
```
cd workspace
sh ./containers/install.sh
```

### Unit test
Unit tests are managed by GoogleTest.
If you add a unit test, modify `test/CMakeLists.txt`.
```
make clean-test
make test-unit
```

### Regression test
- structure: FCC-perturbed TiAl3 (32 atoms)
- potential: gtinv-p=2-l=4-order=2
- 100 timesteps with NVE

```
make test-regression
```

### performance
- perf
```
sudo perf record --call-graph lbr ./../../lmp_mlip_kokkos -in in.mlip
sudo perf report
```

- Flamegraph
```
cd Flamegraph
sudo cp ../perf.data .
sudo perf script -f | ./stackcollapse-perf.pl | ./flamegraph.pl >perf.svg
```
