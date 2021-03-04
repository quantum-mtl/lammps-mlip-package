#!/bin/sh -eu

if [ $# -gt 1 ]; then
    echo "[Usage] install.sh [ncores]"
    exit 1
elif [ $# -eq 1 ]; then
    ncores=$1
else
    ncores=1
fi


lammps_mlip_root="$(dirname $(realpath $0))/.."
lammps_root="${lammps_mlip_root}/lammps"

# copy lammps-mlip
cp -r "${lammps_mlip_root}/lib/mlip" "${lammps_root}/lib"
cp -r "${lammps_mlip_root}/src/USER-MLIP" "${lammps_root}/src"

# modify Makefile
cp "${lammps_mlip_root}/docker/Makefile.lammps" "${lammps_root}/lib/mlip/Makefile.lammps"
cp "${lammps_mlip_root}/docker/Makefile" "${lammps_root}/src/Makefile"
cp "${lammps_mlip_root}/docker/Makefile.mlip_kokkos" "${lammps_root}/src/MAKE/Makefile.mlip_kokkos"

# make
curr="$(pwd)"
cd "${lammps_root}/src"
make yes-kokkos
make yes-manybody
make yes-user-mlip
make ps
make mlip_kokkos -j ${ncores}
cp lmp_mlip_kokkos "${lammps_mlip_root}"
cd "${curr}"
