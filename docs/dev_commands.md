# Commands for development 

## Installation
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

Install pre-commit and formatter
```shell
sudo apt install clang-format
pip install pre-commit
pre-commit install
# pre-commit run --all-files
```

## Formatting

If you want to disable clang-format for some region:
```cpp
// clang-format off
some_codes_here
// clang-format on
```

## Unit test
Unit tests are managed by GoogleTest.
If you add a unit test, modify `test/CMakeLists.txt`.
```
make clean-test
make test-unit
```

## Regression test
- structure: FCC-perturbed TiAl3 (32 atoms)
- potential: gtinv-p=2-l=4-order=2
- 100 timesteps with NVE

```
make test-regression
```

## performance
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
