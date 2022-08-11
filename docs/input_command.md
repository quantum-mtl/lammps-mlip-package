# Lammps input commands to specify a machine learning potential

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
