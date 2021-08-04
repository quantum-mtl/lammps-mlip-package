For OpenMP -- default
```Bash
python3 regression.py --arch OpenMP
```

For Cuda
```Bash
python3 regression.py --arch Cuda
```

| MLP       | des_type | model_type | maxp | order | gtinv_maxl | cutoff | #features |
|-----------|----------|------------|------|-------|------------|--------|-----------|
| pair-17   | pair     | 1          | 3    | 0     | []         | 6.0    | 90        |
| pair-60   | pair     | 2          | 3    | 0     | []         | 12.0   | 22910     |
| gtinv-197 | gtinv    | 2          | 2    | 2     | [4]        | 8.0    | 61605     |
| gtinv-820 | gtinv    | 3          | 3    | 4     | [4, 4, 2]  | 6.0    | 6695      |
| gtinv-434 | gtinv    | 4          | 2    | 3     | [0, 0]     | 8.0    | 5460      |

- structures for triclinic test are generated using [make_lammps_structure_triclinic.ipynb](../../example/triclinic/make_lammps_structure_triclinic.ipynb)
- triclinic test is performed only for gtinv-197 and it compares initial temperature, enegy, and pressure between orthogonal box and triclinic box
- pressure is tested for triclinic box if the difference is less or equal to delta=0.1