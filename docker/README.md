# Run lammps-mlip in docker
```
docker build -t lammps -f docker/Dockerfile .
docker run -it -v $(PWD):/workspace -t lammps
```

in `lammps` container
```
cd /workspace
./docker/install.sh
```
Now `lmp_serial` is built under `/workspace`.
