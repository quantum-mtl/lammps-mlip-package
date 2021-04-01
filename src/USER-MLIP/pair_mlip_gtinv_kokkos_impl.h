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

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::init_style() {
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style mlip_gtinv requires newton pair on");
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
    error->all(FLERR, "Must use half neighbor list style with pair mlip_gtinv/kk");
  }
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
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
                // required when classes are used as functors by Kokkos

  atomKK->sync(Host, datamask_read);
  {
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
      int i = h_ilist(ii);
      int jnum = h_numneigh(i);
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

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::accumulate_energy_and_force_for_all_atom(int inum,
                                                                               int nlocal,
                                                                               int newton_pair,
                                                                               const vector2d &evdwl_array,
                                                                               const vector2d &fx_array,
                                                                               const vector2d &fy_array,
                                                                               const vector2d &fz_array) {
  int i, j, jnum, *jlist;
  double fx, fy, fz, evdwl, dis, delx, dely, delz;
  double **f = atom->f;
  double **x = atom->x;

  double ecoul;
  evdwl = ecoul = 0.0;

  for (int ii = 0; ii < inum; ii++) {
    i = h_ilist(ii);
    jnum = h_numneigh(i);
    for (int jj = 0; jj < jnum; jj++) {
      j = h_neighbors(i, jj);
      j &= NEIGHMASK;
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      dis = sqrt(delx * delx + dely * dely + delz * delz);
      if (dis < pot.get_feature_params().cutoff) {
        evdwl = evdwl_array[ii][jj];
        fx = fx_array[ii][jj];
        fy = fy_array[ii][jj];
        fz = fz_array[ii][jj];
        f[i][0] += fx;
        f[i][1] += fy;
        f[i][2] += fz;
        if (newton_pair || j < nlocal) {
          f[j][0] -= fx;
          f[j][1] -= fy;
          f[j][2] -= fz;
        }
        if (evflag) {
          ev_tally_xyz(i, j, nlocal, newton_pair,
                       evdwl, ecoul, fx, fy, fz, delx, dely, delz);
        }
      }
    }
  }
  if (vflag_fdotr) LAMMPS_NS::Pair::virial_fdotr_compute();
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::compute_energy_and_force_for_each_atom(const barray4dc &prod_anlm_f,
                                                                             const barray4dc &prod_anlm_e,
                                                                             const vector1d &scales,
                                                                             int ii,
                                                                             vector2d &evdwl_array,
                                                                             vector2d &fx_array,
                                                                             vector2d &fy_array,
                                                                             vector2d &fz_array) {
  int i, j, jnum, *jlist, type1, type2, tc, m, lm1, lm2;
  double delx, dely, delz, dis, evdwl, fx, fy, fz,
      costheta, sintheta, cosphi, sinphi, coeff, cc;
  dc f1, ylm_dphi, d0, d1, d2, term1, term2, sume, sumf;

  double **x = atom->x;
  tagint *tag = atom->tag;

  i = h_ilist(ii);
  type1 = types[tag[i] - 1];
  jnum = h_numneigh(i);

  const int n_fn = pot.get_model_params().get_n_fn();
  const int n_des = pot.get_model_params().get_n_des();
  const int n_lm = pot.get_lm_info().size();
  const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;
  const int n_gtinv = pot.get_model_params().get_linear_term_gtinv().size();

  vector1d fn, fn_d;
  vector1dc ylm, ylm_dtheta;
  vector2dc fn_ylm, fn_ylm_dx, fn_ylm_dy, fn_ylm_dz;

  fn_ylm = fn_ylm_dx = fn_ylm_dy = fn_ylm_dz
      = vector2dc(n_fn, vector1dc(n_lm_all));

  for (int jj = 0; jj < jnum; jj++) {
    j = h_neighbors(i, jj);
    j &= NEIGHMASK;
    delx = x[i][0] - x[j][0];
    dely = x[i][1] - x[j][1];
    delz = x[i][2] - x[j][2];
    dis = sqrt(delx * delx + dely * dely + delz * delz);

    if (dis < pot.get_feature_params().cutoff) {
      type2 = types[tag[j] - 1];

      const vector1d &sph
          = cartesian_to_spherical(vector1d{delx, dely, delz});
      get_fn(dis, pot.get_feature_params(), fn, fn_d);
      get_ylm(sph, pot.get_lm_info(), ylm, ylm_dtheta);

      costheta = cos(sph[0]), sintheta = sin(sph[0]);
      cosphi = cos(sph[1]), sinphi = sin(sph[1]);
      fabs(sintheta) > 1e-15 ?
      (coeff = 1.0 / sintheta) : (coeff = 0);
      for (int lm = 0; lm < n_lm; ++lm) {
        m = pot.get_lm_info()[lm][1], lm1 = pot.get_lm_info()[lm][2],
        lm2 = pot.get_lm_info()[lm][3];
        cc = pow(-1, m);
        ylm_dphi = dc{0.0, 1.0} * double(m) * ylm[lm];
        term1 = ylm_dtheta[lm] * costheta;
        term2 = coeff * ylm_dphi;
        d0 = term1 * cosphi - term2 * sinphi;
        d1 = term1 * sinphi + term2 * cosphi;
        d2 = -ylm_dtheta[lm] * sintheta;
        for (int n = 0; n < n_fn; ++n) {
          fn_ylm[n][lm1] = fn[n] * ylm[lm];
          fn_ylm[n][lm2] = cc * std::conj(fn_ylm[n][lm1]);
          f1 = fn_d[n] * ylm[lm];
          fn_ylm_dx[n][lm1] = -(f1 * delx + fn[n] * d0) / dis;
          fn_ylm_dx[n][lm2] = cc * std::conj(fn_ylm_dx[n][lm1]);
          fn_ylm_dy[n][lm1] = -(f1 * dely + fn[n] * d1) / dis;
          fn_ylm_dy[n][lm2] = cc * std::conj(fn_ylm_dy[n][lm1]);
          fn_ylm_dz[n][lm1] = -(f1 * delz + fn[n] * d2) / dis;
          fn_ylm_dz[n][lm2] = cc * std::conj(fn_ylm_dz[n][lm1]);
        }
      }

      const int tc0 = get_type_comb()[type1][type2];
      const auto &prodif = prod_anlm_f[tc0][tag[i] - 1];
      const auto &prodie = prod_anlm_e[tc0][tag[i] - 1];
      const auto &prodjf = prod_anlm_f[tc0][tag[j] - 1];
      const auto &prodje = prod_anlm_e[tc0][tag[j] - 1];

      evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
      // including polynomial correction
      for (int n = 0; n < n_fn; ++n) {
        for (int lm0 = 0; lm0 < n_lm_all; ++lm0) {
          sume = prodie[n][lm0] + prodje[n][lm0] * scales[lm0];
          sumf = prodif[n][lm0] + prodjf[n][lm0] * scales[lm0];
          evdwl += prod_real(fn_ylm[n][lm0], sume);
          fx += prod_real(fn_ylm_dx[n][lm0], sumf);
          fy += prod_real(fn_ylm_dy[n][lm0], sumf);
          fz += prod_real(fn_ylm_dz[n][lm0], sumf);
        }
      }
      evdwl_array[ii][jj] = evdwl;
      fx_array[ii][jj] = fx;
      fy_array[ii][jj] = fy;
      fz_array[ii][jj] = fz;
    }
  }
}

/// @param[out] prod_anlm_f
/// @param[out] prod_anlm_e
template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::compute_partial_anlm_product_for_each_atom(const int n_fn,
                                                                                 const int n_lm_all,
                                                                                 const barray4dc &anlm,
                                                                                 int ii,
                                                                                 barray4dc &prod_anlm_f,
                                                                                 barray4dc &prod_anlm_e) {
  tagint *tag = atom->tag;
  int i = h_ilist(ii);
  int type1 = types[tag[i] - 1];

  // precompute partial a_nlm product for linear terms
  const int n_gtinv = pot.get_model_params().get_linear_term_gtinv().size();
  const vector2dc &uniq = compute_anlm_uniq_products(type1, anlm[tag[i] - 1]);

  // precompute partial a_nlm product for polynomial terms
  vector1d uniq_p;
  if (pot.get_feature_params().maxp > 1) {
    uniq_p = compute_polynomial_model_uniq_products(type1, anlm[tag[i] - 1], uniq);
  }

  // compute prod_anlm_f and prod_anlm_e
  for (int type2 = 0; type2 < pot.get_feature_params().n_type; ++type2) {
    const int tc0 = get_type_comb()[type1][type2];

    for (int n = 0; n < n_fn; ++n) {
      for (int lm0 = 0; lm0 < n_lm_all; ++lm0) {
        dc sumf(0.0), sume(0.0);
        compute_anlm_linear_term(n_gtinv, tc0, n, lm0, uniq, sumf, sume);
        // polynomial model correction for sumf and sume
        compute_anlm_polynomial_model_correction(uniq_p, tc0, n, lm0, uniq, sumf, sume);

        prod_anlm_f[tc0][tag[i] - 1][n][lm0] = sumf;
        prod_anlm_e[tc0][tag[i] - 1][n][lm0] = sume;
      }
    }
  }
}

/// @brief compute order parameters a_{nlm}
/// @return anlm (inum, n_type, n_fn, n_lm_all)
///         n_type is # of elements. n_fn is # of basis functions.
///         n_lm_all is # of (l, m), equal to (l_max + 1)^2
template<class DeviceType>
barray4dc PairMLIPGtinvKokkos<DeviceType>::compute_anlm() {

  const int n_fn = pot.get_model_params().get_n_fn();
  const int n_lm = pot.get_lm_info().size();
  const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;
  const int n_type = pot.get_feature_params().n_type;

  int inum = list->inum;
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  barray4dc anlm(boost::extents[inum][n_type][n_fn][n_lm_all]);
  barray4d anlm_r(boost::extents[inum][n_type][n_fn][n_lm]);
  barray4d anlm_i(boost::extents[inum][n_type][n_fn][n_lm]);
  std::fill(anlm_r.data(), anlm_r.data() + anlm_r.num_elements(), 0.0);
  std::fill(anlm_i.data(), anlm_i.data() + anlm_i.num_elements(), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
  for (int ii = 0; ii < inum; ii++) {
    int i, j, type1, type2, jnum, sindex, *ilist, *jlist;
    double delx, dely, delz, dis, scale;

    double **x = atom->x;
    tagint *tag = atom->tag;

    i = h_ilist(ii);
    type1 = types[tag[i] - 1];
    jnum = h_numneigh(i);

    vector1d fn;
    vector1dc ylm;
    dc val;
    for (int jj = 0; jj < jnum; ++jj) {
      j = h_neighbors(i, jj);
      j &= NEIGHMASK;
      delx = x[i][0] - x[j][0];
      dely = x[i][1] - x[j][1];
      delz = x[i][2] - x[j][2];
      dis = sqrt(delx * delx + dely * dely + delz * delz);
      if (dis < pot.get_feature_params().cutoff) {
        type2 = types[tag[j] - 1];
        const vector1d &sph
            = cartesian_to_spherical(vector1d{delx, dely, delz});
        get_fn(dis, pot.get_feature_params(), fn);
        get_ylm(sph, pot.get_lm_info(), ylm);
        for (int n = 0; n < n_fn; ++n) {
          for (int lm = 0; lm < n_lm; ++lm) {
            if (pot.get_lm_info()[lm][0] % 2 == 0) scale = 1.0;
            else scale = -1.0;
            val = fn[n] * ylm[lm];
#ifdef _OPENMP
#pragma omp atomic
#endif
            anlm_r[tag[i] - 1][type2][n][lm] += val.real();
#ifdef _OPENMP
#pragma omp atomic
#endif
            anlm_r[tag[j] - 1][type1][n][lm] += val.real() * scale;
#ifdef _OPENMP
#pragma omp atomic
#endif
            anlm_i[tag[i] - 1][type2][n][lm] += val.imag();
#ifdef _OPENMP
#pragma omp atomic
#endif
            anlm_i[tag[j] - 1][type1][n][lm] += val.imag() * scale;
          }
        }
      }
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
  for (int ii = 0; ii < inum; ii++) {
    int m, lm1, lm2;
    double cc;
    for (int type2 = 0; type2 < n_type; ++type2) {
      for (int n = 0; n < n_fn; ++n) {
        for (int lm = 0; lm < n_lm; ++lm) {
          m = pot.get_lm_info()[lm][1];
          lm1 = pot.get_lm_info()[lm][2], lm2 = pot.get_lm_info()[lm][3];
          anlm[ii][type2][n][lm1] =
              {anlm_r[ii][type2][n][lm], anlm_i[ii][type2][n][lm]};
          cc = pow(-1, m);
          anlm[ii][type2][n][lm2] =
              {cc * anlm_r[ii][type2][n][lm],
               -cc * anlm_i[ii][type2][n][lm]};
        }
      }
    }
  }
  return anlm;
}

}
#endif //LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
