## Regression test
- structure: FCC-perturbed TiAl3 (32 atoms)
- potential: gtinv-p=2-l=4-order=2
- 100 timesteps with NVE

```
./test.sh
```


### performance
- perf
```
sudo perf record --call-graph lbr ./../../lmp_serial -in in.mlip
sudo perf report
```

- Flamegraph
```
cd Flamegraph
sudo cp ../perf.data .
sudo perf script -f | ./stackcollapse-perf.pl | ./flamegraph.pl >perf.svg
```
