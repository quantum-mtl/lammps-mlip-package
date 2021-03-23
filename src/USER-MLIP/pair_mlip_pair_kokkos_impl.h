//
// Created by Takayuki Nishiyama on 2021/03/22.
//

#ifndef  LMP_PAIR_MLIP_PAIR_KOKKOS_IMPL_H
#define LMP_PAIR_MLIP_PAIR_KOKKOS_IMPL_H

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neighbor_kokkos.h"
//#include "neigh_list_kokkos.h"
#include "neigh_request.h"
#include "force.h"
#include "comm_kokkos.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"

#include "pair_mlip_pair_kokkos.h"

#define MAXLINE 1024;
#define DELTA 4;

namespace LAMMPS_NS {

template<class DeviceType>
PairMLIPPairKokkos<DeviceType>::PairMLIPPairKokkos(LAMMPS *lmp):PairMLIPPair(lmp) {
  respa_enable = 0;
  single_enable = 0;

  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
//  datamask_read = EMPTY_MASK;
//  datamask_modify = EMPTY_MASK;
//
//  k_cutsq = tdual_fparams("PairMLIPPairKokkos::cutsq", atom->ntypes + 1, atom->ntypes + 1);
//  auto d_cutsq = k_cutsq.template view<DeviceType>();
//  rnd_cutsq = d_cutsq;
  std::cerr << "######################\n";
  std::cerr << "# PairMLIPPairKokkos #\n";
  std::cerr << "######################\n";
}

template<class DeviceType>
PairMLIPPairKokkos<DeviceType>::~PairMLIPPairKokkos() {
  if (copymode) return;
  memoryKK->destroy_kokkos(k_eatom, eatom);
  memoryKK->destroy_kokkos(k_vatom, vatom);
  std::cerr << "#######################\n";
  std::cerr << "# ~PairMLIPPairKokkos #\n";
  std::cerr << "#######################\n";
}
}
#endif //LMP_PAIR_MLIP_PAIR_KOKKOS_IMPL_H
