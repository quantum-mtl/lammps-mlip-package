#!/bin/sh -eu

lammps_mlip_root="$(dirname $(realpath $0))/.."
lammps_root="${lammps_mlip_root}/lammps"
ncores="32"

# copy lammps-mlip
cp -r "${lammps_mlip_root}/lib/mlip" "${lammps_root}/lib"
cp -r "${lammps_mlip_root}/src/USER-MLIP" "${lammps_root}/src"

# modify Makefile
cp "${lammps_mlip_root}/docker/Makefile.lammps" "${lammps_root}/lib/mlip/Makefile.lammps"
cp "${lammps_mlip_root}/docker/Makefile" "${lammps_root}/src/Makefile"

# make
curr="$(pwd)"
cd "${lammps_root}/src"
make yes-manybody
make yes-user-mlip
make ps
make serial -j "${ncores}"
cp lmp_serial "${lammps_mlip_root}"
cd "${curr}"
