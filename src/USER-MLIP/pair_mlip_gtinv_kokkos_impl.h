//
// Created by Takayuki Nishiyama on 2021/04/01.
//

#ifndef LMP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
#define LMP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "force.h"
#include "comm_kokkos.h"
#include "memory_kokkos.h"
#include "error.h"
#include "kokkos.h"

#include "pair_mlip_gtinv_kokkos.h"

#define MAXLINE 1024;
#define DELTA 4;

namespace LAMMPS_NS {

template<class DeviceType>
PairMLIPGtinvKokkos<DeviceType>::PairMLIPGtinvKokkos(LAMMPS *lmp) : PairMLIPGtinv(lmp) {
  respa_enable = 0;
  single_enable = 0;

  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  std::cerr << "#######################\n";
  std::cerr << "# PairMLIPGtinvKokkos #\n";
  std::cerr << "#######################\n";
}

template<class DeviceType>
PairMLIPGtinvKokkos<DeviceType>::~PairMLIPGtinvKokkos() {
  if (copymode) return;
//  memoryKK->destroy_kokkos(k_eatom, eatom);
//  memoryKK->destroy_kokkos(k_vatom, vatom);
  std::cerr << "#######################\n";
  std::cerr << "# ~PairMLIPPairKokkos #\n";
  std::cerr << "#######################\n";
}

//template<class DeviceType>
//void PairMLIPGtinvKokkos<DeviceType>::init_style() {
//  if (force->newton_pair == 0) {
//    error->all(FLERR, "Pair style mlip_gtinv requires newton pair on");
//  }
//
//  neighflag = lmp->kokkos->neighflag;
//  int irequest = neighbor->request(this, instance_me);
//
//  neighbor->requests[irequest]->
//      kokkos_host = Kokkos::Impl::is_same<DeviceType, LMPHostType>::value &&
//      !Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;
//  neighbor->requests[irequest]->
//      kokkos_device = Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;
//
//  if (neighflag == HALF || neighflag == HALFTHREAD) {
//    neighbor->requests[irequest]->full = 0; // 0?
//    neighbor->requests[irequest]->half = 1; // 1?
//  } else {
//    error->all(FLERR, "Must use half neighbor list style with pair mlip_gtinv/kk");
//  }
//}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  eflag = eflag_in;
  vflag = vflag_in;

  atomKK->sync(Host, datamask_read);
  PairMLIPGtinv::compute(eflag, vflag);
  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
}
}
#endif //LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
