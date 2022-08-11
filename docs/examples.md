# Examples

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
