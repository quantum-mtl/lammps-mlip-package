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

  k_cutsq = tdual_fparams("PairMLIPGtinvKokkos::cutsq",atom->ntypes+1,atom->ntypes+1);
  auto d_cutsq = k_cutsq.template view<DeviceType>();
  rnd_cutsq = d_cutsq;

  std::cerr << "#######################\n";
  std::cerr << "# PairMLIPGtinvKokkos #\n";
  std::cerr << "#######################\n";
}

template<class DeviceType>
PairMLIPGtinvKokkos<DeviceType>::~PairMLIPGtinvKokkos() {
  if (copymode) return;
//  memoryKK->destroy_kokkos(k_eatom, eatom);
//  memoryKK->destroy_kokkos(k_vatom, vatom);
  std::cerr << "########################\n";
  std::cerr << "# ~PairMLIPGtinvKokkos #\n";
  std::cerr << "########################\n";
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::coeff(int narg, char **arg) {
  //TODO implement on Device
  if (!allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // read parameter file
  MLIP_NS::read_potential_file(arg[2], ele, mass, &fp, reg_coeffs, gtinvdata);
  model.initialize(fp, reg_coeffs, gtinvdata);

  std::vector<int> map(atom->ntypes);

  for (int i = 3; i < narg; ++i) {
    for (int j = 0; j < ele.size(); ++j) {
      if (strcmp(arg[i], ele[j].c_str()) == 0) {
        map[i - 3] = j;
        break;
      }
    }
  }

  for (MLIP_NS::ElementType i = 1; i <= atom->ntypes; ++i) {
    atom->set_mass(FLERR, i, mass[map[i - 1]]);
    for (MLIP_NS::ElementType j = 1; j <= atom->ntypes; ++j) {
      setflag[i][j] = 1;
    }
  }

  for (MLIP_NS::SiteIdx i = 0; i < atom->natoms; ++i) {
    types.template emplace_back(map[(atom->type)[i] - 1]);
  }
}

template<class DeviceType>
double PairMLIPGtinvKokkos<DeviceType>::init_one(int i, int j) {
  //! DO NOT CALL PairMLIPGtinv::init_one(). It requires pot.get_cutmax(), but we have no `pot`.
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  double cutone = fp.cutoff;
  k_cutsq.h_view(i, j) = k_cutsq.h_view(j, i) = cutone * cutone;
  k_cutsq.template modify<LMPHostType>();
  return cutone;
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::allocate() {
  PairMLIPGtinv::allocate();
  //TODO: implement d_map to execute coeff() on Device
  // int n = atom->ntypes;
  // d_map = Kokkos::View<T_INT*, DeviceType>("PairSNAPKokkos::map",n+1);
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::init_style() {
  if (force->newton_pair == 0) {
    error->all(FLERR, "Pair style mlip_gtinv requires newton pair on");
  }

  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->request(this, instance_me);

  neighbor->requests[irequest]->
      kokkos_host = std::is_same<DeviceType, LMPHostType>::value &&
      !std::is_same<DeviceType, LMPDeviceType>::value;
  neighbor->requests[irequest]->
      kokkos_device = std::is_same<DeviceType, LMPDeviceType>::value;

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

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();

  NeighListKokkos<DeviceType> *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  copymode = 1; // set not to deallocate during destruction
  // required when classes are used as functors by Kokkos

  atomKK->sync(Host, datamask_read);
  { // MLIP_NS compute
    model.set_structure_lmp<PairMLIPGtinvKokkos<DeviceType>, NeighListKokkos<DeviceType>>(this, k_list);
    model.compute();
    // TODO: UPDATE F, Energy, Virial
  }

  if (eflag_global) eng_vdwl += model.get_energy();
  model.get_forces_lmp<PairMLIPGtinvKokkos<DeviceType>>(this);
  if (vflag_fdotr) vflag_fdotr=0;//pair_virial_fdotr_compute(this);
  const std::vector<double> tmp_stress = model.get_stress();
  virial[0] = tmp_stress[0];
  virial[1] = tmp_stress[1];
  virial[2] = tmp_stress[2];
  virial[3] = tmp_stress[3];
  virial[4] = tmp_stress[4];
  virial[5] = tmp_stress[5];

  atomKK->modified(execution_space, datamask_modify);

  copymode = 0;
}


} // namespace LAMMPS

namespace MLIP_NS{
template<class PairStyle, class NeighListKokkos>
void MLIPModel::set_structure_lmp(PairStyle *fpair, NeighListKokkos* k_list) {
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
  inum_ = k_list->inum;
  auto h_x = fpair->atomKK->k_x.view_host();
//  LAMMPS_NS::tagint *tag = fpair->atom->tag;
  auto h_tag = fpair->atomKK->k_tag.view_host();
  const std::vector<ElementType> &types = fpair->types;

  auto h_numneigh = Kokkos::create_mirror_view(d_numneigh);
  auto h_neighbors = Kokkos::create_mirror_view(d_neighbors);
  auto h_ilist = Kokkos::create_mirror_view(d_ilist);
  Kokkos::deep_copy(h_numneigh, d_numneigh);
  Kokkos::deep_copy(h_neighbors, d_neighbors);
  Kokkos::deep_copy(h_ilist, d_ilist);

  // number of (i, j) neighbors
  // TODO: atom-first indexing for GPU
  n_pairs_ = 0;
  for (int ii = 0; ii < inum_; ++ii) {
    const SiteIdx i = h_ilist(ii);
    n_pairs_ += h_numneigh(i);
  }
//  Kokkos::parallel_reduce("number of (i, j) neighbors", inum, KOKKOS_LAMBDA(const int& i, int& lsum){
//    lsum += d_numneigh(i);
//  }, n_pairs_);

  // flatten half-neighbor list
  // neighbor_pair_index and neighbor_pair_displacements
  Kokkos::resize(neighbor_pair_index_kk_, n_pairs_);
  Kokkos::resize(neighbor_pair_displacements_kk_, n_pairs_, 3);
  auto h_neighbor_pair_index = neighbor_pair_index_kk_.view_host();
  auto h_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_host();

  NeighborPairIdx count_neighbor = 0;
  for (int ii = 0; ii < inum_; ++ii) {
    const SiteIdx i = h_ilist(ii);
    const int num_neighbors_i = h_numneigh(i);
    for (int jj = 0; jj < num_neighbors_i; ++jj) {
      SiteIdx j = h_neighbors(i, jj);
      j &= NEIGHMASK;
      const int tagi = h_tag(i) - 1;
      const int tagj = h_tag(j) - 1;
      h_neighbor_pair_index(count_neighbor) = Kokkos::pair<SiteIdx, SiteIdx>(tagi, tagj);
      h_neighbor_pair_displacements(count_neighbor, 0) = h_x(j, 0) - h_x(i, 0);
      h_neighbor_pair_displacements(count_neighbor, 1) = h_x(j, 1) - h_x(i, 1);
      h_neighbor_pair_displacements(count_neighbor, 2) = h_x(j, 2) - h_x(i, 2);
      ++count_neighbor;
    }
  }
  neighbor_pair_index_kk_.modify_host();
  neighbor_pair_displacements_kk_.modify_host();
  neighbor_pair_index_kk_.sync_device();
  neighbor_pair_displacements_kk_.sync_device();

  // neighbor_pair_typecomb
  Kokkos::resize(neighbor_pair_typecomb_kk_, n_pairs_);
  auto h_neighbor_pair_typecomb = neighbor_pair_typecomb_kk_.view_host();
  for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
    const auto &ij = h_neighbor_pair_index(npidx);
    const SiteIdx i = ij.first; // is tag[i] - 1
    const SiteIdx j = ij.second; // is tag[j] - 1
    const ElementType type_i = types[i];
    const ElementType type_j = types[j];
    h_neighbor_pair_typecomb(npidx) = type_pairs_kk_.h_view(type_i, type_j);
  }
  neighbor_pair_typecomb_kk_.modify_host();
  neighbor_pair_typecomb_kk_.sync_device();

  // types
  Kokkos::resize(types_kk_, inum_);
  auto h_types = types_kk_.view_host();
  for (int ii = 0; ii < inum_; ++ii) {
    const SiteIdx i = h_ilist(ii);
    types_kk_.h_view(i) = types[i];
  }
  types_kk_.modify_host();
  types_kk_.sync_device();

  // resize views
  // TODO: move resizes to hide allocation time
  Kokkos::resize(d_distance_, n_pairs_);
  Kokkos::resize(d_fn_, n_pairs_, n_fn_);
  Kokkos::resize(d_fn_der_, n_pairs_, n_fn_);
  Kokkos::resize(d_alp_, n_pairs_, n_lm_half_);
  Kokkos::resize(d_alp_sintheta_, n_pairs_, n_lm_half_);
  Kokkos::resize(d_ylm_, n_pairs_, n_lm_half_);
  Kokkos::resize(d_ylm_dx_, n_pairs_, n_lm_half_);
  Kokkos::resize(d_ylm_dy_, n_pairs_, n_lm_half_);
  Kokkos::resize(d_ylm_dz_, n_pairs_, n_lm_half_);

  Kokkos::resize(d_anlm_r_, inum_, n_types_, n_fn_, n_lm_half_);
  Kokkos::resize(d_anlm_i_, inum_, n_types_, n_fn_, n_lm_half_);
  Kokkos::resize(d_anlm_, inum_, n_types_, n_fn_, n_lm_all_);
  Kokkos::resize(structural_features_kk_, inum_, n_des_);
  Kokkos::resize(d_polynomial_adjoints_, inum_, n_des_);
  Kokkos::resize(d_basis_function_adjoints_, inum_, n_typecomb_, n_fn_, n_lm_half_);

  Kokkos::resize(site_energy_kk_, inum_);
  Kokkos::resize(forces_kk_, inum_, 3);

  Kokkos::fence();
}
template<class PairStyle>
void MLIPModel::get_forces_lmp(PairStyle *fpair) {
  // Kokkos::deep_copy(fpair->f, forces_kk_.d_view); // E:1397 vs 32
//  const auto d_forces = forces_kk_.view_device();
//  Kokkos::parallel_for(
//      "fcopy", range_policy(0, inum_),
//      KOKKOS_LAMBDA(const int i) {
//        fpair->f(i, 0) = d_forces(i, 0);
//        fpair->f(i, 1) = d_forces(i, 1);
//        fpair->f(i, 2) = d_forces(i, 2);
//      });
//  Kokkos::fence();
  forces_kk_.sync_host();
  const auto h_forces = forces_kk_.view_host();
  auto h_f = fpair->atomKK->k_f.view_host();
  auto h_ilist = Kokkos::create_mirror_view(fpair->d_ilist);
  auto h_tag = fpair->atomKK->k_tag.view_host();
  Kokkos::deep_copy(h_ilist, fpair->d_ilist);
  for (SiteIdx ii = 0; ii < inum_; ++ii) {
    const int i = h_ilist(ii);
    const int tagi = h_tag(i)-1;
    h_f(i, 0) = h_forces(tagi, 0);
    h_f(i, 1) = h_forces(tagi, 1);
    h_f(i, 2) = h_forces(tagi, 2);
  }
  fpair->atomKK->k_f.modify_host();
  fpair->atomKK->k_f.sync_device();
}
} // namespace MLIP

#endif //LMP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
