#!/bin/sh -eu

lammps="$(dirname $(realpath $0))/../../lmp_serial"
input="in.mlip"

output="dump.atom"
expect="dump.atom.regression"

${lammps} -in ${input}

if diff -q ${output} ${expect} ; then
  echo "[passed]"
else
  echo "[failed!]"
fi
