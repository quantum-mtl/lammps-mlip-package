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
  vflag = 1;  //!! NASTY HACK !!

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom, vatom);
    memoryKK->create_kokkos(k_vatom, vatom, maxvatom, 6, "pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  NeighListKokkos<DeviceType>* k_list = static_cast<NeighListKokkos<DeviceType>*>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  int inum = list->inum;

  h_numneigh = Kokkos::create_mirror_view(d_numneigh);
  h_neighbors = Kokkos::create_mirror_view(d_neighbors);
  h_ilist = Kokkos::create_mirror_view(d_ilist);
  Kokkos::deep_copy(h_numneigh, d_numneigh);
  Kokkos::deep_copy(h_neighbors, d_neighbors);
  Kokkos::deep_copy(h_ilist, d_ilist);

  copymode = 1; // set not to deallocate during destruction
//  // required when classes are used as functors by Kokkos
//
//  atomKK->sync(execution_space, X_MASK | F_MASK | TYPE_MASK);
//  x = atomKK->k_x.view<DeviceType>();
//  f = atomKK->k_f.view<DeviceType>();
//  type = atomKK->k_type.view<DeviceType>();
//  k_cutsq.template sync<DeviceType>();

  atomKK->sync(Host, datamask_read);
//  PairMLIPPair::compute(eflag, vflag);
  {
    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    const int n_type_comb = pot.get_model_params().get_type_comb_pair().size();
    vector3d prod_all_f(n_type_comb, vector2d(inum));
    vector3d prod_all_e(n_type_comb, vector2d(inum));
    if (pot.get_feature_params().maxp > 1) {
      vector2d dn(inum, vector1d(pot.get_model_params().get_n_des(), 0.0));
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
      for (int ii = 0; ii < inum; ii++) {
        compute_main_structural_feature_for_each_atom(dn, ii);

      }
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
      for (int ii = 0; ii < inum; ii++) {
        compute_partial_structural_feature_for_each_atom(dn, ii, prod_all_f, prod_all_e);
      }
    }

    vector2d evdwl_array(inum), fpair_array(inum);
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
      compute_energy_and_force_for_each_atom(prod_all_f, prod_all_e, ii, evdwl_array, fpair_array);
    }
    accumulate_energy_and_force_for_all_atom(inum, nlocal, newton_pair, evdwl_array, fpair_array);
  }

  atomKK->modified(Host, datamask_modify);
  atomKK->sync(execution_space, datamask_modify);
}

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::init_style() {
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style mlip_pair requires newton pair on");
  }

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->request(this, instance_me);

  neighbor->requests[irequest]->
      kokkos_host = Kokkos::Impl::is_same<DeviceType, LMPHostType>::value &&
      !Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;
  neighbor->requests[irequest]->
      kokkos_device = Kokkos::Impl::is_same<DeviceType, LMPDeviceType>::value;

  if (neighflag == HALF || neighflag == HALFTHREAD) {
    neighbor->requests[irequest]->full = 0; // 0?
    neighbor->requests[irequest]->half = 1; // 1?
  } else {
    error->all(FLERR, "Must use half neighbor list style with pair mlip_pair/kk");
  }
}

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::compute_main_structural_feature_for_each_atom(vector2d &dn, int ii) {
  int i, j, type1, type2, jnum, sindex, *ilist, *jlist;
  double delx, dely, delz, dis;
  double **x = atom->x;
  tagint *tag = atom->tag;

  vector1d fn;
  const int n_fn = pot.get_model_params().get_n_fn();

  i = h_ilist(ii);
  type1 = types[tag[i] - 1];
  jnum = h_numneigh[i];
//  jlist = list->firstneigh[i];
  for (int jj = 0; jj < jnum; jj++) {
    j = h_neighbors(i, jj);//jlist[jj];
    j &= NEIGHMASK;
    delx = x[i][0] - x[j][0];
    dely = x[i][1] - x[j][1];
    delz = x[i][2] - x[j][2];
    dis = sqrt(delx * delx + dely * dely + delz * delz);

    if (dis < pot.get_feature_params().cutoff) {
      type2 = types[tag[j] - 1];
      sindex = get_type_comb()[type1][type2] * n_fn;
      get_fn(dis, pot.get_feature_params(), fn);
      for (int n = 0; n < n_fn; ++n) {
#ifdef _OPENMP
#pragma omp atomic
#endif
        dn[tag[i] - 1][sindex + n] += fn[n];
#ifdef _OPENMP
#pragma omp atomic
#endif
        dn[tag[j] - 1][sindex + n] += fn[n];
      }
    }
  }
}

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::compute_partial_structural_feature_for_each_atom(const vector2d &dn,
                                                                                      int ii,
                                                                                      vector3d &prod_all_f,
                                                                                      vector3d &prod_all_e) {
  int i, *ilist, type1;
  double **x = atom->x;
  tagint *tag = atom->tag;
  ilist = list->ilist;

  i = h_ilist[ii], type1 = types[tag[i] - 1];
  const int n_fn = pot.get_model_params().get_n_fn();
  const vector1d &prodi
      = polynomial_model_uniq_products(dn[tag[i] - 1]);
  for (int type2 = 0; type2 < pot.get_feature_params().n_type; ++type2) {
    const int tc = get_type_comb()[type1][type2];
    vector1d vals_f(n_fn, 0.0), vals_e(n_fn, 0.0);
    for (int n = 0; n < n_fn; ++n) {
      double v;
      for (const auto &pi:
          pot.get_poly_feature().get_polynomial_info(tc, n)) {
        v = prodi[pi.comb_i] * pot.get_reg_coeffs()[pi.reg_i];
        vals_f[n] += v;
        vals_e[n] += v / pi.order;
      }
    }
    prod_all_f[tc][tag[i] - 1] = vals_f;
    prod_all_e[tc][tag[i] - 1] = vals_e;
  }
}

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::compute_energy_and_force_for_each_atom(const vector3d &prod_all_f,
                                                                            const vector3d &prod_all_e,
                                                                            int ii,
                                                                            vector2d &evdwl_array,
                                                                            vector2d &fpair_array) {
  int i, j, jnum, *jlist, type1, type2, sindex, tc;
  double delx, dely, delz, dis, fpair, evdwl;
  double **x = atom->x;
  tagint *tag = atom->tag;

  i = h_ilist[ii];
  type1 = types[tag[i] - 1];
  jnum = h_numneigh[i];
//  jlist = list->firstneigh[i];

  const int n_fn = pot.get_model_params().get_n_fn();
  vector1d fn, fn_d;

  evdwl_array[ii].resize(jnum);
  fpair_array[ii].resize(jnum);
  for (int jj = 0; jj < jnum; jj++) {
    j = h_neighbors(i, jj);//jlist[jj];
    j &= NEIGHMASK;
    delx = x[i][0] - x[j][0];
    dely = x[i][1] - x[j][1];
    delz = x[i][2] - x[j][2];
    dis = sqrt(delx * delx + dely * dely + delz * delz);

    if (dis < pot.get_feature_params().cutoff) {
      type2 = types[tag[j] - 1];
      tc = get_type_comb()[type1][type2];
      sindex = tc * n_fn;

      get_fn(dis, pot.get_feature_params(), fn, fn_d);
      fpair = dot(fn_d, pot.get_reg_coeffs(), sindex);
      evdwl = dot(fn, pot.get_reg_coeffs(), sindex);

      // polynomial model correction
      if (pot.get_feature_params().maxp > 1) {
        fpair += dot(fn_d, prod_all_f[tc][tag[i] - 1], 0)
            + dot(fn_d, prod_all_f[tc][tag[j] - 1], 0);
        evdwl += dot(fn, prod_all_e[tc][tag[i] - 1], 0)
            + dot(fn, prod_all_e[tc][tag[j] - 1], 0);
      }
      // polynomial model correction: end

      fpair *= -1.0 / dis;
      evdwl_array[ii][jj] = evdwl;
      fpair_array[ii][jj] = fpair;
    }
  }
}

template<class DeviceType>
void PairMLIPPairKokkos<DeviceType>::accumulate_energy_and_force_for_all_atom(int inum, int nlocal, int newton_pair,
                                                                              const vector2d &evdwl_array,
                                                                              const vector2d &fpair_array) {
  int i, j, jnum, *jlist;
  double fpair, evdwl, dis, delx, dely, delz;
  double **f = atom->f;
  double **x = atom->x;

  for (int ii = 0; ii < inum; ii++) {
    i = h_ilist[ii];
    jnum = h_numneigh[i];
//    jlist = list->firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
      j = h_neighbors(i, jj);//jlist[jj];
      j &= NEIGHMASK;
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      dis = sqrt(delx * delx + dely * dely + delz * delz);
      if (dis < pot.get_feature_params().cutoff) {
        evdwl = evdwl_array[ii][jj];
        fpair = fpair_array[ii][jj];
        f[i][0] += fpair * delx;
        f[i][1] += fpair * dely;
        f[i][2] += fpair * delz;
        //            if (newton_pair || j < nlocal)
        f[j][0] -= fpair * delx;
        f[j][1] -= fpair * dely;
        f[j][2] -= fpair * delz;
        if (evflag) {
          ev_tally(i, j, nlocal, newton_pair,
                   evdwl, 0.0, fpair, delx, dely, delz);
        }
      }
    }
  }
}
}
#endif //LMP_PAIR_MLIP_PAIR_KOKKOS_IMPL_H
