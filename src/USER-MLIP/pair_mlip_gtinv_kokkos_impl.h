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
  {
    vflag = 1;
    if (eflag || vflag) {
      ev_setup(eflag, vflag);
    } else {
      evflag = 0;
    }

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    const int n_type_comb = pot.get_model_params().get_type_comb_pair().size(); // equal to n_type * (n_type + 1) / 2
    const int n_fn = pot.get_model_params().get_n_fn();
    const int n_lm = pot.get_lm_info().size();
    const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;

    // order paramter, (inum, n_type, n_fn, n_lm_all)
    const barray4dc &anlm = compute_anlm();

    // partial a_nlm products
    barray4dc prod_anlm_f(boost::extents[n_type_comb][inum][n_fn][n_lm_all]);
    barray4dc prod_anlm_e(boost::extents[n_type_comb][inum][n_fn][n_lm_all]);
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
      compute_partial_anlm_product_for_each_atom(n_fn, n_lm_all, anlm, ii, prod_anlm_f, prod_anlm_e);
    }

    vector2d evdwl_array(inum), fx_array(inum), fy_array(inum), fz_array(inum);
    for (int ii = 0; ii < inum; ii++) {
      int i = list->ilist[ii];
      int jnum = list->numneigh[i];
      evdwl_array[ii].resize(jnum);
      fx_array[ii].resize(jnum);
      fy_array[ii].resize(jnum);
      fz_array[ii].resize(jnum);
    }

    vector1d scales;
    for (int l = 0; l < pot.get_feature_params().maxl + 1; ++l) {
      if (l % 2 == 0) for (int m = -l; m < l + 1; ++m) scales.emplace_back(1.0);
      else for (int m = -l; m < l + 1; ++m) scales.emplace_back(-1.0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
      compute_energy_and_force_for_each_atom(prod_anlm_f, prod_anlm_e, scales, ii, evdwl_array,
                                             fx_array, fy_array, fz_array);
    }

    accumulate_energy_and_force_for_all_atom(inum, nlocal, newton_pair, evdwl_array, fx_array, fy_array, fz_array);
  }
  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
}
}
#endif //LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
