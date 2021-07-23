//
// Created by Takayuki Nishiyama on 2021/04/01.
//

#ifndef LMP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
#define LMP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "pair_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "neighbor.h"
#include "neigh_list_kokkos.h"
#include "neighbor_kokkos.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
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
  // reference(stable) PairEAMAlloyKokkos)
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_kokkos.cpp#L38
  Pair::respa_enable = 0;
  Pair::one_coeff = 1;
  Pair::manybody_flag = 1;
  // Pair::kokkosable = 1;  // is not exist in stable.

  Pointers::atomKK = (AtomKokkos *) atom;
  Pair::execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  Pair::datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  Pair::datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;

  model = nullptr;
  fp = nullptr;
  inum = eflag = vflag = neighflag = need_dup = 0;

  std::cerr << "#######################\n";
  std::cerr << "# PairMLIPGtinvKokkos #\n";
  std::cerr << "#######################\n";
}

template<class DeviceType>
PairMLIPGtinvKokkos<DeviceType>::~PairMLIPGtinvKokkos() {
  // reference(stable) PairEAMAlloyKokkos
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L58
  // reference(stable) PairEAM
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/MANYBODY/pair_eam.cpp#L76-L140
  if(!copymode) {
    memoryKK->destroy_kokkos(k_eatom, eatom);
    memoryKK->destroy_kokkos(k_vatom, vatom);

    if (model) {
      delete model;
      model = nullptr;
    }
    if (fp) {
      delete fp;
      fp = nullptr;
    }
    std::cerr << "########################\n";
    std::cerr << "# ~PairMLIPGtinvKokkos #\n";
    std::cerr << "########################\n";
  }
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::settings(int narg, char **arg){
  if (narg > 0) error->all(FLERR,"Illegal pair_style command");
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::coeff(int narg, char **arg) {
  // Do not call the FU**ING parent class's coeff()
  // TODO implement on Device
  // reference(stable) PairEAMAlloyKokkos
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L912

  if (!Pair::allocated) allocate();

  if (narg != 3 + atom->ntypes)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // insure I,J args are * *
  if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
    error->all(FLERR,"Incorrect args for pair coefficients");

  // Read potential file and initialize MLIPModel.
  // To avoid, delete them if exist.
  {
    // read parameter file
    if (fp) {
      delete fp;
      fp = nullptr;
    }
    fp = new MLIP_NS::MLIPInput();
    MLIP_NS::read_potential_file(arg[2], ele, mass, fp, reg_coeffs, gtinvdata);

    // initialize MLIP model instance
    if (model) {
      delete model;
      model = nullptr;
    }
    model = new MLIP_NS::MLIPModelLMP<PairMLIPGtinvKokkos<DeviceType>, NeighListKokkos<DeviceType>>();
    model->initialize(*fp, reg_coeffs, gtinvdata, this);
  }

  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L940-L952
  // read args that map atom types to elements in potential file
  // map[i] = which element the Ith atom type is, -1 if "NULL"
  {
    int i, j;
    for (i = 3; i < narg; i++) {
      if (strcmp(arg[i], "NULL") == 0) {
        map[i - 2] = -1;
        continue;
      }
      for (j = 0; j < ele.size(); j++)
        if (strcmp(arg[i], ele[j].c_str()) == 0) break;
      if (j < ele.size()) map[i - 2] = j;
      else error->all(FLERR, "No matching element in Machine Learning Interatomic Potential file");
    }
  }

  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L954-L959
  // clear setflag since coeff() called once with I,J = * *
  {
    MLIP_NS::ElementType i, j;
    MLIP_NS::ElementType n = atom->ntypes;
    for (i = 1; i <= n; i++)
      for (j = i; j <= n; j++)
        Pair::setflag[i][j] = 0;
  }

  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L961-L975
  // set setflag i,j for type pairs where both are mapped to elements
  // set mass of atom type if i = j
  {
    int count = 0;
    MLIP_NS::ElementType i, j;
    MLIP_NS::ElementType n = atom->ntypes;
    for (i = 1; i <= n; i++) {
      for (j = i; j <= n; j++) {
        if (map[i] >= 0 && map[j] >= 0) {
          Pair::setflag[i][j] = 1;
          if (i == j) atom->set_mass(FLERR, i, mass[map[i]]);
          count++;
        }
      }
    }
    if (count == 0) error->all(FLERR, "Incorrect args for pair coefficients");
  }

  // copy LAMMPS_NS::Atom::type to LAMMPS_NS::PairMLIPGtinv::types
  // map[i] starts from i=1
  {
    MLIP_NS::SiteIdx i;
    MLIP_NS::SiteIdx n = atom->natoms;
    if (!types.empty()) {
      types.clear();
    }
    for (i = 0; i < n; ++i) {
      types.template emplace_back(map[(atom->type)[i]]); // atom->type[] and map[] starts from 0 and 1, respectively
    }
  }
}

template<class DeviceType>
double PairMLIPGtinvKokkos<DeviceType>::init_one(int i, int j) {
  //! DO NOT CALL PairMLIPGtinv::init_one(). It requires pot.get_cutmax(), but we have no `pot`.
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  double cutone = fp->cutoff;
  return cutone;
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::allocate() {
  PairMLIPGtinv::allocate();
  map.resize(atom->ntypes + 1);
  //TODO: implement d_map to execute coeff() on Device
  // int n = atom->ntypes;
  // d_map = Kokkos::View<T_INT*, DeviceType>("PairSNAPKokkos::map",n+1);
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::init_style() {
  // parent class doesn't have init_style()...
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/MANYBODY/pair_tersoff.cpp#L366
  if (atom->tag_enable == 0)
    error->all(FLERR,"Pair style mlip_gtinv requires atom IDs");
  // if (force->newton_pair == 0) {
  //   error->all(FLERR, "Pair style mlip_gtinv requires newton pair on");
  // }

  Pair::init_style(); // just request neighbor

  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_tersoff_kokkos.cpp#L81
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L294
  neighflag = lmp->kokkos->neighflag;
  int irequest = neighbor->nrequest - 1;

  neighbor->requests[irequest]->
      kokkos_host = std::is_same<DeviceType, LMPHostType>::value &&
      !std::is_same<DeviceType, LMPDeviceType>::value;
  neighbor->requests[irequest]->
      kokkos_device = std::is_same<DeviceType, LMPDeviceType>::value;

  // Check flags:
  // Only 'neigh full newton off' is allowed for MPI parallelization.
  // Without MPI parallelization, 'neigh full newton off' and 'neigh half newton on' are allowed.
  int nprocs;
  MPI_Comm_size(world,&nprocs);
  newton_pair = force->newton_pair;
  if (nprocs==1){  // Without MPI
    if (neighflag == FULL) {  // neigh full
      if (newton_pair == 0) {  // newton off
        neighbor->requests[irequest]->full = 1;
        neighbor->requests[irequest]->half = 0;
      } else {  // newton on
        error->all(FLERR, "Must use 'neigh half newton on' or 'neigh full newton off' with mlip_gtinv/kk");
      }
    } else if (neighflag == HALF || neighflag == HALFTHREAD) {  // neigh half
      if (newton_pair == 0) {  // newton off
        error->all(FLERR, "Must use 'neigh half newton on' or 'neigh full newton off' with mlip_gtinv/kk");
      } else {  // newton on
        neighbor->requests[irequest]->full = 0;
        neighbor->requests[irequest]->half = 1;
      }
    } else {
      error->all(FLERR, "Must use 'neigh half newton on' or 'neigh full newton off' with mlip_gtinv/kk");
    }
  } else {  // With MPI
    if (neighflag == FULL) {  // neigh full
      if (newton_pair == 0) {  // newton off
        neighbor->requests[irequest]->full = 1;
        neighbor->requests[irequest]->half = 0;
      } else {  // newton on
        error->all(FLERR, "Cannot (yet) use 'neigh full newton on' with MPI parallelization for mlip_gtinv/kk");
      }
    } else if (neighflag == HALF || neighflag == HALFTHREAD) {  // neigh half
      error->all(FLERR, "Cannot (yet) use neigh half with MPI parallelization for mlip_gtinv/kk");
    }
  }
}

template<class DeviceType>
void PairMLIPGtinvKokkos<DeviceType>::compute(int eflag_in, int vflag_in) {
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;
  no_virial_fdotr_compute = 1;

  ev_init(eflag, vflag, 0);

  // From pair_lj_cut_kokkos.cpp
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_lj_cut_kokkos.cpp#L87-L98
  {
    // reallocate per-atom arrays if necessary

    if (eflag_atom) {
      memoryKK->destroy_kokkos(k_eatom, eatom);
      memoryKK->create_kokkos(k_eatom, eatom, maxeatom, "pair:eatom");
      d_eatom = k_eatom.view<DeviceType>();
    }
    if (vflag_atom) {
      memoryKK->destroy_kokkos(k_vatom, vatom);
      memoryKK->create_kokkos(k_vatom, vatom, maxvatom, "pair:vatom");
      d_vatom = k_vatom.view<DeviceType>();
    }
  }

  // From pair_lj_cut_kokkos.cpp
  // Sync to execution space before compute pair.
  // Set modify_flag before compute pair.
  // https://github.com/lammps/lammps/blob/584943fc928351bc29f41a132aee3586e0a2286a/src/KOKKOS/pair_lj_cut_kokkos.cpp#L100-L104
  {
    atomKK->sync(execution_space, datamask_read);
    k_cutsq.template sync<DeviceType>();
//    k_params.template sync<DeviceType>();
    if (eflag || vflag) atomKK->modified(execution_space, datamask_modify);
    else atomKK->modified(execution_space, F_MASK);
  }

  x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();

  auto *k_list = static_cast<NeighListKokkos<DeviceType> *>(list);
  d_numneigh = k_list->d_numneigh;
  d_neighbors = k_list->d_neighbors;
  d_ilist = k_list->d_ilist;
  inum = list->inum;

  // From pair_eam_alloy_kokkos.cpp
  // Assign dupulicated views when HALFTHREAD and not using atomics.
  // https://github.com/lammps/lammps/blob/998b76520e74b3a90580bf1a92155dcbe2843dba/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L124-L135
  {
    need_dup = lmp->kokkos->need_dup<DeviceType>();
    if (need_dup) {
      dup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(f);
      dup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_eatom);
      dup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterDuplicated>(d_vatom);
    } else {
      ndup_f     = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(f);
      ndup_eatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_eatom);
      ndup_vatom = Kokkos::Experimental::create_scatter_view<Kokkos::Experimental::ScatterSum, Kokkos::Experimental::ScatterNonDuplicated>(d_vatom);
    }
  }

  copymode = 1; // set not to deallocate during destruction
  // required when classes are used as functors by Kokkos

  atomKK->sync(Host, datamask_read);

  model->set_structure(this, k_list);
  model->compute(k_list);
  // model->get_forces<PairMLIPGtinvKokkos<DeviceType>>(this);

  // From pair_eam_alloy_kokkos.cpp
  // https://github.com/lammps/lammps/blob/998b76520e74b3a90580bf1a92155dcbe2843dba/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L251-L252
  {
    if (need_dup)
      Kokkos::Experimental::contribute(f, dup_f);
  }

  if (eflag_global) eng_vdwl += model->get_energy();
  if (vflag_global) {
    const std::vector<double> tmp_stress = model->get_stress();
    virial[0] = tmp_stress[0];
    virial[1] = tmp_stress[1];
    virial[2] = tmp_stress[2];
    virial[3] = tmp_stress[3];
    virial[4] = tmp_stress[4];
    virial[5] = tmp_stress[5];
  }

  // From pair_eam_alloy_kokkos.cpp
  // https://github.com/lammps/lammps/blob/998b76520e74b3a90580bf1a92155dcbe2843dba/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L264-L276
  {
    if (eflag_atom) {
      if (need_dup)
        Kokkos::Experimental::contribute(d_eatom, dup_eatom);
      k_eatom.template modify<DeviceType>();
      k_eatom.template sync<LMPHostType>();
    }

    if (vflag_atom) {
      if (need_dup)
        Kokkos::Experimental::contribute(d_vatom, dup_vatom);
      k_vatom.template modify<DeviceType>();
      k_vatom.template sync<LMPHostType>();
    }
  }

  if (vflag_fdotr && !no_virial_fdotr_compute) pair_virial_fdotr_compute(this);
  copymode = 0;

  // From pair_eam_alloy_kokkos.cpp
  // https://github.com/lammps/lammps/blob/998b76520e74b3a90580bf1a92155dcbe2843dba/src/KOKKOS/pair_eam_alloy_kokkos.cpp#L282-L288
  {// free duplicated memory
    if (need_dup) {
      dup_f = decltype(dup_f)();
      dup_eatom = decltype(dup_eatom)();
      dup_vatom = decltype(dup_vatom)();
    }
  }
}

} // namespace LAMMPS

namespace MLIP_NS{
template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::set_structure(PairStyle *fpair, NeighListKokkos *k_list) {
  auto d_numneigh = k_list->d_numneigh;
  auto d_neighbors = k_list->d_neighbors;
  auto d_ilist = k_list->d_ilist;
  inum_ = k_list->inum;  // Number of I atoms neighbors are stored for
  nlocal_ = fpair->atomKK->nlocal;  // Number of owned atoms in this proc.
  nall_ = fpair->atomKK->nlocal + fpair->atomKK->nghost;  // Total number of owned and ghost atoms on this proc
  auto h_x = fpair->atomKK->k_x.view_host();
//  LAMMPS_NS::tagint *tag = fpair->atom->tag;
  auto h_tag = fpair->atomKK->k_tag.view_host();
  const std::vector<ElementType> &types = fpair->types;

  fpair->atomKK->k_x.sync_host();
  fpair->atomKK->k_tag.sync_host();

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
    const LMPLocalIdx i = h_ilist(ii);
    n_pairs_ += h_numneigh(i);
  }
//  Kokkos::parallel_reduce("number of (i, j) neighbors", inum, KOKKOS_LAMBDA(const int& i, int& lsum){
//    lsum += d_numneigh(i);
//  }, n_pairs_);

  // flatten half-neighbor list
  // neighbor_pair_index and neighbor_pair_displacements
  Kokkos::realloc(neighbor_pair_index_kk_, n_pairs_);
//  neighbor_pair_index_kk_.sync_host();
  Kokkos::realloc(neighbor_pair_displacements_kk_, n_pairs_, 3);
//  neighbor_pair_displacements_kk_.sync_host();
  auto h_neighbor_pair_index = neighbor_pair_index_kk_.view_host();
  auto h_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_host();

  // Get neighbor indices.
  // For full neighbor list, indices are stored as LMPLocalIdx.
  // For half neighbor list, indices are stored as LMPAtomID (=tagint-1).
  NeighborPairIdx count_neighbor = 0;
  if (neighflag_ == FULL) {
    for (int ii = 0; ii < inum_; ++ii) {
      const LMPLocalIdx i = h_ilist(ii);
      // Store pair index centered on owned atom.
      // i is restricted to owned atom.
      // j can be greater than nlocal_, i.e. ghost atom.
      if (i < nlocal_) {
        const int num_neighbors_i = h_numneigh(i);
        for (int jj = 0; jj < num_neighbors_i; ++jj) {
          LMPLocalIdx j = h_neighbors(i, jj);
          j &= NEIGHMASK;
          h_neighbor_pair_index(count_neighbor) = Kokkos::pair<LMPLocalIdx, LMPLocalIdx>(i, j);
          ++count_neighbor;
        }
      }
    }
  } else if (neighflag_ == HALF || neighflag_ == HALFTHREAD) {
    for (int ii = 0; ii < inum_; ++ii) {
      const LMPLocalIdx i = h_ilist(ii);
      const int num_neighbors_i = h_numneigh(i);
      const LMPAtomID site_i = h_tag(i) - 1;
      for (int jj = 0; jj < num_neighbors_i; ++jj) {
        LMPLocalIdx j = h_neighbors(i, jj);
        j &= NEIGHMASK;
        const LMPAtomID site_j = h_tag(j) - 1;
        h_neighbor_pair_index(count_neighbor) = Kokkos::pair<LMPAtomID, LMPAtomID>(site_i, site_j);
        ++count_neighbor;
      }
    }
  }

  // Store pair displacements.
  // Displacements are calculated in local index, i.e. including ghost atoms.
  count_neighbor = 0;
  for (int ii = 0; ii < inum_; ++ii) {
    const LMPLocalIdx i = h_ilist(ii);
//    const LAMMPS_NS::AtomNeighborsConst neighbors_i = k_list->get_neighbors_const(i);
    if (i < nlocal_) {
      const int num_neighbors_i = h_numneigh(i);
      const X_FLOAT xtmp = h_x(i, 0);
      const X_FLOAT ytmp = h_x(i, 1);
      const X_FLOAT ztmp = h_x(i, 2);
      for (int jj = 0; jj < num_neighbors_i; ++jj) {
        LMPLocalIdx j = h_neighbors(i, jj);
//      SiteIdx j = neighbors_i(jj);
        j &= NEIGHMASK;
        h_neighbor_pair_displacements(count_neighbor, 0) = h_x(j, 0) - xtmp;
        h_neighbor_pair_displacements(count_neighbor, 1) = h_x(j, 1) - ytmp;
        h_neighbor_pair_displacements(count_neighbor, 2) = h_x(j, 2) - ztmp;
        ++count_neighbor;
      }
    }
  }
  neighbor_pair_index_kk_.modify_host();
  neighbor_pair_displacements_kk_.modify_host();
  neighbor_pair_index_kk_.sync_device();
  neighbor_pair_displacements_kk_.sync_device();

  // neighbor_pair_typecomb
  Kokkos::realloc(neighbor_pair_typecomb_kk_, n_pairs_);
//  neighbor_pair_typecomb_kk_.sync_host();
  auto h_neighbor_pair_typecomb = neighbor_pair_typecomb_kk_.view_host();
  if (neighflag_ == FULL) {
    for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
      const auto &ij = h_neighbor_pair_index(npidx);
      const LMPAtomID i = h_tag(ij.first) - 1;
      const LMPAtomID j = h_tag(ij.second) - 1;
      const ElementType type_i = types[i];
      const ElementType type_j = types[j];
      h_neighbor_pair_typecomb(npidx) = type_pairs_kk_.h_view(type_i, type_j);
    }
  } else if (neighflag_ == HALF || neighflag_ == HALFTHREAD) {
    for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
      const auto &ij = h_neighbor_pair_index(npidx);
      const LMPAtomID i = ij.first;
      const LMPAtomID j = ij.second;
      const ElementType type_i = types[i];
      const ElementType type_j = types[j];
      h_neighbor_pair_typecomb(npidx) = type_pairs_kk_.h_view(type_i, type_j);
    }
  }
  neighbor_pair_typecomb_kk_.modify_host();
  neighbor_pair_typecomb_kk_.sync_device();

  // types
  if (neighflag_ == FULL) {
    Kokkos::realloc(types_kk_, nall_);
    auto h_types = types_kk_.view_host();
    for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
      const auto &ij = h_neighbor_pair_index(npidx);
      const LMPLocalIdx i = ij.first;
      const LMPLocalIdx j = ij.second;
      const LMPAtomID site_i = h_tag(i) - 1;
      const LMPAtomID site_j = h_tag(j) - 1;
      h_types(i) = types[site_i];
      h_types(j) = types[site_j];
    }
  } else if (neighflag_ == HALF || neighflag_ == HALFTHREAD) {
    Kokkos::realloc(types_kk_, inum_);
    auto h_types = types_kk_.view_host();
    for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
      const auto &ij = h_neighbor_pair_index(npidx);
      const LMPAtomID site_i = ij.first;
      const LMPAtomID site_j = ij.second;
      h_types(site_i) = types[site_i];
      h_types(site_j) = types[site_j];
    }
  }
    types_kk_.modify_host();
  types_kk_.sync_device();

  // resize views
  // TODO: move resizes to hide allocation time
  Kokkos::realloc(d_distance_, n_pairs_);
  Kokkos::realloc(d_fn_, n_pairs_, n_fn_);
  Kokkos::realloc(d_fn_der_, n_pairs_, n_fn_);
  Kokkos::realloc(d_alp_, n_pairs_, n_lm_half_);
  Kokkos::realloc(d_alp_sintheta_, n_pairs_, n_lm_half_);
  Kokkos::realloc(d_ylm_, n_pairs_, n_lm_half_);
  Kokkos::realloc(d_ylm_dx_, n_pairs_, n_lm_half_);
  Kokkos::realloc(d_ylm_dy_, n_pairs_, n_lm_half_);
  Kokkos::realloc(d_ylm_dz_, n_pairs_, n_lm_half_);

  Kokkos::realloc(d_anlm_r_, nlocal_, n_types_, n_fn_, n_lm_half_);
  Kokkos::realloc(d_anlm_i_, nlocal_, n_types_, n_fn_, n_lm_half_);
  Kokkos::realloc(d_anlm_, nlocal_, n_types_, n_fn_, n_lm_all_);
  Kokkos::realloc(structural_features_kk_, nlocal_, n_des_);
//  structural_features_kk_.sync_host();
  Kokkos::realloc(d_polynomial_adjoints_, nlocal_, n_des_);
  Kokkos::realloc(d_basis_function_adjoints_, nlocal_, n_typecomb_, n_fn_, n_lm_half_);

  Kokkos::realloc(site_energy_kk_, nlocal_);
//  site_energy_kk_.sync_host();
  Kokkos::realloc(forces_kk_, nlocal_, 3);
//  forces_kk_.sync_host();

  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::get_forces(PairStyle *fpair) {
  forces_kk_.sync_host();
  fpair->atomKK->k_f.sync_host();
  const auto h_forces = forces_kk_.view_host();
  auto h_f = fpair->atomKK->k_f.view_host();
  auto h_neighbor_pair_index = neighbor_pair_index_kk_.view_host();
  for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
    const auto &ij = h_neighbor_pair_index(npidx);
    const LMPLocalIdx i = ij.first;
    const LMPLocalIdx j = ij.second;
    h_f(i, 0) = h_forces(i, 0);
    h_f(i, 1) = h_forces(i, 1);
    h_f(i, 2) = h_forces(i, 2);
    h_f(j, 0) = h_forces(j, 0);
    h_f(j, 1) = h_forces(j, 1);
    h_f(j, 2) = h_forces(j, 2);
  }
  fpair->atomKK->k_f.modify_host();
  fpair->atomKK->k_f.sync_device();
  Kokkos::fence();
}

}  // MLIP_NS
#endif //LMP_PAIR_MLIP_GTINV_KOKKOS_IMPL_H_
