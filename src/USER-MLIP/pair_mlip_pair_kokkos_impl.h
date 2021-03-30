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
  // datamask_read = X_MASK | V_MASK | F_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | ENERGY_MASK | VIRIAL_MASK;
  // datamask_modify = X_MASK | V_MASK | F_MASK | TAG_MASK | TYPE_MASK | MASK_MASK | ENERGY_MASK | VIRIAL_MASK;

  k_cutsq = tdual_fparams("PairMLIPPairKokkos::cutsq", atom->ntypes + 1, atom->ntypes + 1);
  auto d_cutsq = k_cutsq.template view<DeviceType>();
  rnd_cutsq = d_cutsq;
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

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  eflag = eflag_in;
  vflag = vflag_in;

//  if (neighflag == FULL) no_virial_fdotr_compute = 1;
//
//  ev_init(eflag, vflag, 0);
//
//  // reallocate per-atom arrays if necessary
//
//  if (eflag_atom) {
//std::cerr << "ef" ;
//    memoryKK->destroy_kokkos(k_eatom, eatom);
//    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
//    d_eatom = k_eatom.view<DeviceType>();
//  }
//  if (vflag_atom) {
//std::cerr << "vf";
//    memoryKK->destroy_kokkos(k_vatom, vatom);
//    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, 6, "pair:vatom");
//    d_vatom = k_vatom.view<DeviceType>();
//  }
//
//  copymode = 1; // set not to deallocate during destruction
//  // required when classes are used s functors by Kokkos
//
//  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
//  x = atomKK->k_x.view<DeviceType>();
//  f = atomKK->k_f.view<DeviceType>();
//  type = atomKK->k_type.view<DeviceType>();
//  k_cutsq.template sync<DeviceType>();

  // inum = list->inum;

//  for (size_t ii = 0; ii != inum; ++ii) {
//    atomKK->k_x.h_view(ii, 0) = atom->x[ii][0];
//    atomKK->k_x.h_view(ii, 1) = atom->x[ii][1];
//    atomKK->k_x.h_view(ii, 2) = atom->x[ii][2];
//    atomKK->k_f.h_view(ii, 0) = atom->f[ii][0];
//    atomKK->k_f.h_view(ii, 1) = atom->f[ii][1];
//    atomKK->k_f.h_view(ii, 2) = atom->f[ii][2];
//    atomKK->k_v.h_view(ii, 0) = atom->v[ii][0];
//    atomKK->k_v.h_view(ii, 1) = atom->v[ii][1];
//    atomKK->k_v.h_view(ii, 2) = atom->v[ii][2];
//    atomKK->k_mass.h_view(ii) = atom->mass[ii];
//    atomKK->k_tag.h_view(ii) = atom->tag[ii];
//    atomKK->k_type.h_view(ii) = atom->type[ii];
//    atomKK->k_mask.h_view(ii) = atom->mask[ii];
//    atomKK->k_image.h_view(ii) = atom->image[ii];
//  }
//  atomKK->sync(execution_space, ALL_MASK);
//  atomKK->k_mass.modify_host();
//  atomKK->k_mass.sync_device();
//lmp->kokkos->auto_sync = 1;
//atomKK->modified(execution_space, datamask_read);
atomKK->sync(Host, datamask_read);
  PairMLIPPair::compute(eflag, vflag);

//  for (size_t ii = 0; ii != inum; ++ii) {
//    atomKK->k_x.h_view(ii, 0) = atom->x[ii][0];
//    atomKK->k_x.h_view(ii, 1) = atom->x[ii][1];
//    atomKK->k_x.h_view(ii, 2) = atom->x[ii][2];
//    atomKK->k_f.h_view(ii, 0) = atom->f[ii][0];
//    atomKK->k_f.h_view(ii, 1) = atom->f[ii][1];
//    atomKK->k_f.h_view(ii, 2) = atom->f[ii][2];
//    atomKK->k_v.h_view(ii, 0) = atom->v[ii][0];
//    atomKK->k_v.h_view(ii, 1) = atom->v[ii][1];
//    atomKK->k_v.h_view(ii, 2) = atom->v[ii][2];
//    atomKK->k_mass.h_view(ii) = atom->mass[ii];
//    atomKK->k_tag.h_view(ii) = atom->tag[ii];
//    atomKK->k_type.h_view(ii) = atom->type[ii];
//    atomKK->k_mask.h_view(ii) = atom->mask[ii];
//    atomKK->k_image.h_view(ii) = atom->image[ii];
//  }
// atomKK->modified(Host, ALL_MASK);
//   atomKK->sync(execution_space, ALL_MASK);
atomKK->modified(Host, datamask_modify);
atomKK->sync(execution_space, datamask_modify);

//  for (size_t ii = 0; ii != list->inum; ++ii) {
//    atomKK->k_mass.h_view(ii) = atom->mass[ii];
//  }
//  atomKK->k_mass.modify_host();
//  atomKK->k_mass.sync_device();
//  atomKK->k_mass.modify_host();
//  atomKK->k_mass.sync_device();
//  atomKK->k_x.modify_host();
//  atomKK->k_x.sync_device();
//  atomKK->k_f.modify_host();
//  atomKK->k_f.sync_device();
//  atomKK->k_v.modify_host();
//  atomKK->k_v.sync_device();
//  atomKK->k_tag.modify_host();
//  atomKK->k_tag.sync_device();
}

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::init_style() {
std::cerr << "PairMLIPPairKokkos::init_style";
  PairMLIPPair::init_style();
//  if (force->newton_pair == 0) {
//    error->all(FLERR, "Pair style mlip_pair requires newton pair on");
//  }
//
//  neighflag = lmp->kokkos->neighflag;
//  int irequest = neighbor->request(this, instance_me);
//
//  neighbor->requests[irequest]->kokkos_host =
//      Kokkos::Impl::is_same<DeviceType, LMPHostType>::value &&
//          !Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;
//  neighbor->requests[irequest]->kokkos_device =
//      Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;
//
//  if (neighflag == HALF || neighflag == HALFTHREAD) {
//    neighbor->requests[irequest]->full = 0;
//    neighbor->requests[irequest]->half = 1;
//  } else {
//    error->all(FLERR, "Must use half neighbor list style with pair mlip_pair/kk");
//  }
}
}
#endif //LMP_PAIR_MLIP_PAIR_KOKKOS_IMPL_H
