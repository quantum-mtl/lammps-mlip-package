# Python interface

## Installation with CMake

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
      -D CMAKE_BUILD_TYPE=Release \
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

## Simple usages

```shell
cd exmaples/
python3
```

```python
from lammps import PyLammps
# load `liblammps_{name}.so`
cmdargs = ['-log', 'none', '-kokkos', 'on', 'gpus', '1', '-suffix', 'kk', '-pk', 'kokkos', 'neigh', 'full', 'newton', 'on']
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
