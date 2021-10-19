//
// Created on 2021/07/16.
//

#ifndef LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_MLIPKK_LMP_H_
#define LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_MLIPKK_LMP_H_

#include "mlipkk.h"
#include "kokkos_type.h"

namespace MLIP_NS {
using LMPLocalIdx = int;  // locally assigned indices for atoms on this proc
using LMPAtomID = int;  // LAMMPS atom ID, i.e. tagint-1.
using ArgIdxPairCoeff = int;            // Index for arguments of `pair_coeff`
using ElementIdxInFile = int;           // Index for elements in potential file
using ElementNameInFile = std::string;  // Element name stored in potential file
using NumOfAtomsInSystem = int;         // # of atoms in the system, i.e. atom->natoms
using NumOfAtomsOnEachProc = int;       // # of atoms on the proc, owned + ghost
using NumOfElementType = int;           // # of element types.

// For overloading compute functions.
// Use as a flag.
struct NeighFull {};
struct NeighHalf {};
struct NeighHalfThread {};
struct NewtonOn {};
struct NewtonOff {};

template<class PairStyle, class NeighListKokkos>
class MLIPModelLMP : public MLIPModel {
 public:
  MLIPModelLMP() = default;
  ~MLIPModelLMP() = default;

  void initialize(const MLIPInput &input, const vector1d &reg_coeffs, const Readgtinv &gtinvdata, PairStyle *fpair);
  void compute(NeighListKokkos *k_list);

  void compute_order_parameters(NeighListKokkos *k_list);
  void compute_order_parameters(NeighFull neighflag, NewtonOff newton_pair, NeighListKokkos *k_list);
  void compute_order_parameters(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_order_parameters(NeighHalfThread neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_order_parameters(NeighHalf neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);

  void compute_structural_features(NeighListKokkos *k_list);
  void compute_structural_features(NeighFull neighflag, NewtonOff newton_pair, NeighListKokkos *k_list);
  void compute_structural_features(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_structural_features(NeighHalfThread neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_structural_features(NeighHalf neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);

  void compute_energy(NeighListKokkos *k_list);
  void compute_energy(NeighFull neighflag, NewtonOff newton_pair, NeighListKokkos *k_list);
  void compute_energy(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_energy(NeighHalfThread neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_energy(NeighHalf neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);

  void compute_polynomial_adjoints(NeighListKokkos *k_list);
  void compute_polynomial_adjoints(NeighFull neighflag, NewtonOff newton_pair, NeighListKokkos *k_list);
  void compute_polynomial_adjoints(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_polynomial_adjoints(NeighHalfThread neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_polynomial_adjoints(NeighHalf neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);

  void compute_basis_function_adjoints(NeighListKokkos *k_list);
  void compute_basis_function_adjoints(NeighFull neighflag, NewtonOff newton_pair, NeighListKokkos *k_list);
  void compute_basis_function_adjoints(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_basis_function_adjoints(NeighHalfThread neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_basis_function_adjoints(NeighHalf neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);

  void compute_forces_and_stress(NeighListKokkos *k_list);
  void compute_forces_and_stress(NeighFull neighflag, NewtonOff newton_pair, NeighListKokkos *k_list);
  void compute_forces_and_stress(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_forces_and_stress(NeighHalfThread neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);
  void compute_forces_and_stress(NeighHalf neighflag, NewtonOn newton_pair, NeighListKokkos *k_list);

  // defined here for LAMMPS interface
  void set_structure(PairStyle *fpair, NeighListKokkos* k_list);

  // defined here for LAMMPS interface
  void get_forces(PairStyle *fpair, NeighListKokkos *k_list);
  vector1d get_stress();

 protected:
  /* Total number of owned and ghost atoms on this proc*/
  int nall_;
  /* Number of owned atoms in this proc. */
  int nlocal_;
  /* Type of neighbor list, FULL, HALF, or HALFTHREAD. */
  int neighflag_;
  /* Newton's 3rd law setting: 0 for off, 1 for on. */
  int newton_pair_;

  NeighFull neighfull_;
  NeighHalf neighhalf_;
  NeighHalfThread neighhalfthread_;
  NewtonOn newtonon_;
  NewtonOff newtonoff_;
};
}  // MLIP_NS

/// template implementations
namespace MLIP_NS {

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::initialize(const MLIPInput &input,
                                                          const vector1d &reg_coeffs,
                                                          const Readgtinv &gtinvdata,
                                                          PairStyle *fpair) {
  MLIPModel::initialize(input, reg_coeffs, gtinvdata);
  nall_ = 0;
  neighflag_ = fpair->lmp->kokkos->neighflag;
  newton_pair_ = fpair->force->newton_pair;
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute(NeighListKokkos *k_list) {
#ifdef _DEBUG
  auto end = std::chrono::system_clock::now();
    std::chrono::nanoseconds::rep time_bf = 0;
    std::chrono::nanoseconds::rep time_op = 0;
    std::chrono::nanoseconds::rep time_sf = 0;
    std::chrono::nanoseconds::rep time_energy = 0;
    std::chrono::nanoseconds::rep time_pa = 0;
    std::chrono::nanoseconds::rep time_ba = 0;
    std::chrono::nanoseconds::rep time_acc = 0;
#endif

  compute_basis_functions();

#ifdef _DEBUG
  time_bf += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Basis    : " << time_bf << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_order_parameters(k_list);

#ifdef _DEBUG
  time_op += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "anlm     : " << time_op << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_structural_features(k_list);

#ifdef _DEBUG
  time_sf += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "features : " << time_sf << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_energy(k_list);

#ifdef _DEBUG
  time_energy += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Energy   : " << time_energy << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_polynomial_adjoints(k_list);

#ifdef _DEBUG
  time_pa += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Poly Ad  : " << time_pa << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_basis_function_adjoints(k_list);

#ifdef _DEBUG
  time_ba += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Basis Ad : " << time_ba << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_forces_and_stress(k_list);

#ifdef _DEBUG
  time_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Acc      : " << time_acc << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_order_parameters(NeighListKokkos *k_list) {
  if (neighflag_ == FULL && newton_pair_ == 0) {
    compute_order_parameters(neighfull_, newtonoff_, k_list);
  } else if (neighflag_ == FULL && newton_pair_ == 1) {
    compute_order_parameters(neighfull_, newtonon_, k_list);
  } else if (neighflag_ == HALF) {
    compute_order_parameters(neighhalf_, newtonon_, k_list);
  } else if (neighflag_ == HALFTHREAD) {
    compute_order_parameters(neighhalfthread_, newtonon_, k_list);
  }
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_structural_features(NeighListKokkos *k_list) {
  if (neighflag_ == FULL && newton_pair_ == 0) {
    compute_structural_features(neighfull_, newtonoff_, k_list);
  } else if (neighflag_ == FULL && newton_pair_ == 1) {
    compute_structural_features(neighfull_, newtonon_, k_list);
  } else if (neighflag_ == HALF) {
    compute_structural_features(neighhalf_, newtonon_, k_list);
  } else if (neighflag_ == HALFTHREAD) {
    compute_structural_features(neighhalfthread_, newtonon_, k_list);
  }
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_energy(NeighListKokkos *k_list) {
  if (neighflag_ == FULL && newton_pair_ == 0) {
    compute_energy(neighfull_, newtonoff_, k_list);
  } else if (neighflag_ == FULL && newton_pair_ == 1) {
    compute_energy(neighfull_, newtonon_, k_list);
  } else if (neighflag_ == HALF) {
    compute_energy(neighhalf_, newtonon_, k_list);
  } else if (neighflag_ == HALFTHREAD) {
    compute_energy(neighhalfthread_, newtonon_, k_list);
  }
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_polynomial_adjoints(NeighListKokkos *k_list) {
  if (neighflag_ == FULL && newton_pair_ == 0) {
    compute_polynomial_adjoints(neighfull_, newtonoff_, k_list);
  } else if (neighflag_ == FULL && newton_pair_ == 1) {
    compute_polynomial_adjoints(neighfull_, newtonon_, k_list);
  } else if (neighflag_ == HALF) {
    compute_polynomial_adjoints(neighhalf_, newtonon_, k_list);
  } else if (neighflag_ == HALFTHREAD) {
    compute_polynomial_adjoints(neighhalfthread_, newtonon_, k_list);
  }
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_basis_function_adjoints(NeighListKokkos *k_list) {
  if (neighflag_ == FULL && newton_pair_ == 0) {
    compute_basis_function_adjoints(neighfull_, newtonoff_, k_list);
  } else if (neighflag_ == FULL && newton_pair_ == 1) {
    compute_basis_function_adjoints(neighfull_, newtonon_, k_list);
  } else if (neighflag_ == HALF) {
    compute_basis_function_adjoints(neighhalf_, newtonon_, k_list);
  } else if (neighflag_ == HALFTHREAD) {
    compute_basis_function_adjoints(neighhalfthread_, newtonon_, k_list);
  }
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_forces_and_stress(NeighListKokkos *k_list) {
  if (neighflag_ == FULL && newton_pair_ == 0) {
    compute_forces_and_stress(neighfull_, newtonoff_, k_list);
  } else if (neighflag_ == FULL && newton_pair_ == 1) {
    compute_forces_and_stress(neighfull_, newtonon_, k_list);
  } else if (neighflag_ == HALF) {
    compute_forces_and_stress(neighhalf_, newtonon_, k_list);
  } else if (neighflag_ == HALFTHREAD) {
    compute_forces_and_stress(neighhalfthread_, newtonon_, k_list);
  }
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_order_parameters(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list) {
  Kokkos::parallel_for("init_anlm_full",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {nlocal_, n_types_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const LMPLocalIdx i, const ElementType type, const int n, const LMInfoIdx lmi) {
                         d_anlm_r_(i, type, n, lmi) = 0.0;
                         d_anlm_i_(i, type, n, lmi) = 0.0;
                       }
  );

  const auto d_types = types_kk_.view_device();
  const auto d_neighbor_pair_index = neighbor_pair_index_kk_.view_device();
  const auto d_lm_info = lm_info_kk_.view_device();
  sview_4d s_anlm_r (d_anlm_r_);
  sview_4d s_anlm_i (d_anlm_i_);
  // const auto d_ilist = k_list->d_ilist;

  // compute order paramters for m <= 0
  Kokkos::parallel_for("anlm_full",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {n_pairs_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx, const int n, const LMInfoIdx lmi) {
                         const auto& ij = d_neighbor_pair_index(npidx);
                         const LMPLocalIdx i = ij.first;
                         const LMPLocalIdx j = ij.second;

//                         const ElementType type_i = d_types(i);
                         const ElementType type_j = d_types(j);

                         auto s_anlm_r_a = s_anlm_r.access();
                         auto s_anlm_i_a = s_anlm_i.access();

                         const int l = d_lm_info(lmi, 0);
//                         const double scale = (l % 2) ? -1.0 : 1.0;  // sign for parity
                         const Kokkos::complex<double> val = d_fn_(npidx, n) * d_ylm_(npidx, lmi);
                         // neighbors_ is a full list!!!
                         s_anlm_r_a(i, type_j, n, lmi) += val.real();
                         s_anlm_i_a(i, type_j, n, lmi) -= val.imag();  // take c.c.
                        //  if (i != j) {
                        //    s_anlm_r_a(j, type_i, n, lmi) += val.real() * scale;
                        //    s_anlm_i_a(j, type_i, n, lmi) -= val.imag() * scale;  // take c.c
                        //  }
                       }
  );
  Kokkos::Experimental::contribute(d_anlm_r_, s_anlm_r);
  Kokkos::Experimental::contribute(d_anlm_i_, s_anlm_i);

  // augment order paramters for m > 0
  Kokkos::parallel_for("anlm_all",
                       Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {nlocal_, n_types_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const LMPLocalIdx i, const ElementType type, const int n, const LMInfoIdx lmi) {
                         const int m = d_lm_info(lmi, 1);
                         const LMIdx lm1 = d_lm_info(lmi, 2);  // idx for (l, m)
                         const LMIdx lm2 = d_lm_info(lmi, 3);  // idx for (l, -m)
                         d_anlm_(i, type, n, lm1) = Kokkos::complex<double>(d_anlm_r_(i, type, n, lmi),
                                                                            d_anlm_i_(i, type, n, lmi));
                         double cc = (m % 2) ? -1.0 : 1.0;  // sign for complex conjugate
                         d_anlm_(i, type, n, lm2) = Kokkos::complex<double>(cc * d_anlm_r_(i, type, n, lmi),
                                                                            - cc * d_anlm_i_(i, type, n, lmi));
                       }
  );

  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_order_parameters(NeighFull neighflag,
                                                                        NewtonOff newton_pair,
                                                                        NeighListKokkos *k_list) {
  ;  // do nothing
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_order_parameters(NeighHalfThread neighflag,
                                                                        NewtonOn newton_pair,
                                                                        NeighListKokkos *k_list) {
  MLIPModel::compute_order_parameters();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_order_parameters(NeighHalf neighflag,
                                                                        NewtonOn newton_pair,
                                                                        NeighListKokkos *k_list) {
  MLIPModel::compute_order_parameters();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_structural_features(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list) {
  const auto d_types = types_kk_.view_device();
  const auto d_other_type = other_type_kk_.view_device();
  const auto d_irreps_type_intersection = irreps_type_intersection_.view_device();
  const auto d_irreps_type_mapping = irreps_type_mapping_.view_device();
  const auto d_irreps_first_term = irreps_first_term_.view_device();
  const auto d_irreps_num_terms = irreps_num_terms_.view_device();
  const auto d_lm_coeffs = lm_coeffs_kk_.view_device();
  auto d_structural_features = structural_features_kk_.view_device();

  const auto d_ilist = k_list->d_ilist;

  Kokkos::parallel_for("init_structural_features",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nlocal_, n_des_}),
                       KOKKOS_CLASS_LAMBDA(const LMPLocalIdx i, const FeatureIdx fidx) {
                         d_structural_features(i, fidx) = 0.0;
                       }
  );

  Kokkos::parallel_for("structural_features",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, n_des_}),
                       KOKKOS_CLASS_LAMBDA(const int ii, const FeatureIdx fidx) {
                         const IrrepsTypeCombIdx itcidx = fidx % n_irreps_typecomb_;  // should be consistent with poly_.get_irreps_type_idx

                         const LMPLocalIdx i = d_ilist(ii);
                         const ElementType type_i = d_types(i);
                         if (!d_irreps_type_intersection(itcidx, type_i)) {
                           // not related element type
                           return;
                         }

                         auto type_combs_rowview = d_irreps_type_combs_.rowConst(itcidx);
                         const int n = fidx / n_irreps_typecomb_;
                         const IrrepsIdx iidx = d_irreps_type_mapping(itcidx);
                         const IrrepsTermIdx first_term = d_irreps_first_term(iidx);
                         const int num_terms = d_irreps_num_terms(iidx);

                         double feature = 0.0;
                         for (int term = 0; term < num_terms; ++term) {
                           const IrrepsTermIdx iterm = first_term + term;
                           auto lm_term = d_lm_array_.rowConst(iterm);
                           Kokkos::complex<double> tmp = d_lm_coeffs(iterm);
                           for (int p = 0; p < type_combs_rowview.length; ++p) {
                             const TypeCombIdx tcidx = type_combs_rowview(p);
                             const ElementType type2 = d_other_type(tcidx, type_i);
                             const LMIdx lm = lm_term(p);
                             tmp *= d_anlm_(i, type2, n, lm);
                           }
                           feature += tmp.real();
                         }
                         d_structural_features(i, fidx) = feature;
                       }
  );
  structural_features_kk_.modify_device();
  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_structural_features(NeighFull neighflag,
                                                                           NewtonOff newton_pair,
                                                                           NeighListKokkos *k_list) {
  ;  // do nothing
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_structural_features(NeighHalfThread neighflag,
                                                                           NewtonOn newton_pair,
                                                                           NeighListKokkos *k_list) {
  MLIPModel::compute_structural_features();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_structural_features(NeighHalf neighflag,
                                                                           NewtonOn newton_pair,
                                                                           NeighListKokkos *k_list) {
  MLIPModel::compute_structural_features();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_energy(NeighFull neighflag, NewtonOn newton_pair, NeighListKokkos *k_list) {
  const int num_poly_idx = n_reg_coeffs_;
  auto d_site_energy = site_energy_kk_.view_device();
  auto d_ilist = k_list->d_ilist;

  // initialize
  Kokkos::parallel_for("init_energy",
                       range_policy(0, nlocal_),
                       KOKKOS_LAMBDA(const LMPLocalIdx i) {
                         d_site_energy(i) = 0.0;
                       }
  );

  const auto d_reg_coeffs = reg_coeffs_kk_.view_device();
  const auto d_structural_features = structural_features_kk_.view_device();

  // energy for each atom-i
  sview_1d sd_site_energy(d_site_energy);
  Kokkos::parallel_for("site_energy",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, num_poly_idx}),
                       KOKKOS_CLASS_LAMBDA(const int ii, const PolynomialIdx pidx) {
                         const LMPLocalIdx i = d_ilist(ii);
                         double feature = 1.0;
                         auto rowView = d_polynomial_index_.rowConst(pidx);
                         for (int ffidx = 0; ffidx < rowView.length; ++ffidx) {
                           const FeatureIdx fidx = rowView(ffidx);
                           feature *= d_structural_features(i, fidx);
                         }
                         auto sd_site_energy_a = sd_site_energy.access();
                         sd_site_energy_a(i) += d_reg_coeffs(pidx) * feature;
                       }
  );
  Kokkos::Experimental::contribute(d_site_energy, sd_site_energy);
  site_energy_kk_.modify_device();

  // total energy
  double energy = 0.0;
  Kokkos::parallel_reduce("energy",
                          range_policy(0, inum_),
                          KOKKOS_CLASS_LAMBDA(const int ii, double& energy_tmp) {
                            const LMPLocalIdx i = d_ilist(ii);
                            energy_tmp += site_energy_kk_.d_view(i);
                          },
                          energy
  );
  Kokkos::fence();
  energy_ = energy;

  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_energy(NeighFull neighflag,
                                                              NewtonOff newton_pair,
                                                              NeighListKokkos *k_list) {
  ;  // do nothing
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_energy(NeighHalfThread neighflag,
                                                              NewtonOn newton_pair,
                                                              NeighListKokkos *k_list) {
  MLIPModel::compute_energy();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_energy(NeighHalf neighflag,
                                                              NewtonOn newton_pair,
                                                              NeighListKokkos *k_list) {
  MLIPModel::compute_energy();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_polynomial_adjoints(NeighFull neighflag,
                                                                           NewtonOn newton_pair,
                                                                           NeighListKokkos *k_list) {
  const int num_poly_idx = n_reg_coeffs_;

  const auto d_reg_coeffs = reg_coeffs_kk_.view_device();
  const auto d_structural_features = structural_features_kk_.view_device();
  sview_2d sd_polynomial_adjoints(d_polynomial_adjoints_);

  const auto d_ilist = k_list->d_ilist;

  Kokkos::parallel_for("init_polynomial_adjoints",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nlocal_, n_des_}),
                       KOKKOS_CLASS_LAMBDA(const LMPLocalIdx i, const FeatureIdx fidx) {
                         d_polynomial_adjoints_(i, fidx) = 0.0;
                       }
  );

  // scratch memory for d_structural_features(i, :)
  using ScratchPadView = Kokkos::View<double *, ExecSpace::scratch_memory_space>;
  size_t scratch_bytes = ScratchPadView::shmem_size(n_des_);
  const int scratch_level = 0;

  Kokkos::parallel_for("polynomial_adjoints",
                       team_policy(inum_, Kokkos::AUTO).set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes)),
                       KOKKOS_CLASS_LAMBDA(const team_policy::member_type &teamMember) {
                         const SiteIdx ii = teamMember.league_rank();
                         const LMPLocalIdx i = d_ilist(ii);

                         // load into scratch
                         const ScratchPadView d_structural_features_i(teamMember.team_scratch(scratch_level), n_des_);
                         Kokkos::parallel_for(
                             Kokkos::TeamVectorRange(teamMember, n_des_),
                             [=, *this](const FeatureIdx fidx) {
                               d_structural_features_i(fidx) = d_structural_features(i, fidx);
                             }
                         );
                         teamMember.team_barrier();

                         Kokkos::parallel_for(
                             Kokkos::TeamThreadRange(teamMember, num_poly_idx),
                             [=, *this](const PolynomialIdx pidx) {
                               auto sd_polynomial_adjoints_a = sd_polynomial_adjoints.access();
                               auto rowView = d_polynomial_index_.rowConst(pidx);
                               const int poly_order = rowView.length;
                               for (int p1 = 0; p1 < poly_order; ++p1) {
                                 double adjoint = d_reg_coeffs(pidx);
                                 for (int p2 = 0; p2 < poly_order; ++p2) {
                                   if (p2 == p1) {
                                     continue;
                                   }
                                   const FeatureIdx fidx2 = rowView(p2);
                                   adjoint *= d_structural_features_i(fidx2);
                                 }

                                 const FeatureIdx fidx1 = rowView(p1);
                                 sd_polynomial_adjoints_a(i, fidx1) += adjoint;
                               }
                             }
                         );
                       }
  );
  Kokkos::Experimental::contribute(d_polynomial_adjoints_, sd_polynomial_adjoints);
  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_polynomial_adjoints(NeighFull neighflag,
                                                                           NewtonOff newton_pair,
                                                                           NeighListKokkos *k_list) {
  ;  // do nothing
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_polynomial_adjoints(NeighHalfThread neighflag,
                                                                           NewtonOn newton_pair,
                                                                           NeighListKokkos *k_list) {
  MLIPModel::compute_polynomial_adjoints();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_polynomial_adjoints(NeighHalf neighflag,
                                                                           NewtonOn newton_pair,
                                                                           NeighListKokkos *k_list) {
  MLIPModel::compute_polynomial_adjoints();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_basis_function_adjoints(NeighFull neighflag,
                                                                               NewtonOn newton_pair,
                                                                               NeighListKokkos *k_list) {
  const auto d_types = types_kk_.view_device();
  const auto d_other_type = other_type_kk_.view_device();
  const auto d_irreps_type_intersection = irreps_type_intersection_.view_device();
  const auto d_irreps_type_mapping = irreps_type_mapping_.view_device();
  const auto d_irreps_num_terms = irreps_num_terms_.view_device();
  const auto d_irreps_first_term = irreps_first_term_.view_device();
  const auto d_lm_coeffs = lm_coeffs_kk_.view_device();
  const auto d_lm2l = lm2l_.view_device();
  const auto d_lm2m = lm2m_.view_device();
  sview_4dc s_basis_function_adjoints(d_basis_function_adjoints_);
  const auto d_ilist = k_list->d_ilist;

  Kokkos::parallel_for("init_basis_function_adjoints",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0},
                                                                         {nlocal_, n_typecomb_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const LMPLocalIdx i, const TypeCombIdx tcidx, const int n, const LMInfoIdx lmi) {
                         d_basis_function_adjoints_(i, tcidx, n, lmi) = 0.0;
                       }
  );

  Kokkos::parallel_for("basis_function_adjoints",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {inum_, n_irreps_typecomb_, n_fn_}),
                       KOKKOS_CLASS_LAMBDA(const SiteIdx ii, const IrrepsTypeCombIdx itcidx, const int n) {
                         const LMPLocalIdx i = d_ilist(ii);
                         const ElementType type_i = d_types(i);
                         if (!d_irreps_type_intersection(itcidx, type_i)) {
                           // not related type
                           return;
                         }

                         auto type_combs_rowview = d_irreps_type_combs_.rowConst(itcidx);
                         const IrrepsIdx iidx = d_irreps_type_mapping(itcidx);
                         const IrrepsTermIdx first_term = d_irreps_first_term(iidx);
                         const int num_terms = d_irreps_num_terms(iidx);
                         const int order = type_combs_rowview.length;

                         auto sh_basis_function_adjoints_a = s_basis_function_adjoints.access();

                         const FeatureIdx
                             fidx = n * n_irreps_typecomb_ + itcidx;  // consistent with poly_.get_feature_idx
                         for (int term = 0; term < num_terms; ++term) {
                           const IrrepsTermIdx iterm = first_term + term;
                           const auto cg = d_lm_coeffs(iterm);
                           auto lm_term = d_lm_array_.rowConst(iterm);
                           for (int mu = 0; mu < order; ++mu) {
                             const LMIdx lm_mu = lm_term(mu);
                             const int m_mu = d_lm2m(lm_mu);
                             // if magnetic quantum number `m` of lm_mu is positive, not needed
                             if (m_mu > 0) {
                               continue;
                             }
                             const int l_mu = d_lm2l(lm_mu);
                             const LMInfoIdx lmi_mu = l_mu * (l_mu + 1) / 2 + l_mu + m_mu;

                             Kokkos::complex<double> tmp = cg * d_polynomial_adjoints_(i, fidx);
                             for (int mu2 = 0; mu2 < order; ++mu2) {
                               if (mu2 == mu) {
                                 continue;
                               }
                               const TypeCombIdx tcidx_mu2 = type_combs_rowview(mu2);
                               const ElementType type_mu2 = d_other_type(tcidx_mu2, type_i);
                               const LMIdx lm_mu2 = lm_term(mu2);
                               tmp *= d_anlm_(i, type_mu2, n, lm_mu2);
                             }

                             const TypeCombIdx tc_mu = type_combs_rowview(mu);
                             sh_basis_function_adjoints_a(i, tc_mu, n, lmi_mu) += tmp;
                           }
                         }
                       }
  );
  Kokkos::Experimental::contribute(d_basis_function_adjoints_, s_basis_function_adjoints);

  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_basis_function_adjoints(NeighFull neighflag,
                                                                               NewtonOff newton_pair,
                                                                               NeighListKokkos *k_list) {
  ;  // do nothing
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_basis_function_adjoints(NeighHalfThread neighflag,
                                                                               NewtonOn newton_pair,
                                                                               NeighListKokkos *k_list) {
  MLIPModel::compute_basis_function_adjoints();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_basis_function_adjoints(NeighHalf neighflag,
                                                                               NewtonOn newton_pair,
                                                                               NeighListKokkos *k_list) {
  MLIPModel::compute_basis_function_adjoints();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_forces_and_stress(NeighFull neighflag,
                                                                         NewtonOn newton_pair,
                                                                         NeighListKokkos *k_list) {
  auto d_forces = forces_kk_.view_device();
  auto d_stress = stress_kk_.view_device();

  // initialize
  Kokkos::parallel_for("init_forces",
                       Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {nlocal_, 3}),
                       KOKKOS_LAMBDA(const SiteIdx i, const int x) {
                         d_forces(i, x) = 0.0;
                       }
  );
  Kokkos::parallel_for("init_stress",
                       range_policy(0, 6),
                       KOKKOS_LAMBDA(const int vi) {
                         d_stress(vi) = 0.0;
                       }
  );

  const auto d_lm_info = lm_info_kk_.view_device();
  const auto d_neighbor_pair_index = neighbor_pair_index_kk_.view_device();
  const auto d_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_device();
  const auto d_neighbor_pair_typecomb = neighbor_pair_typecomb_kk_.view_device();

  sview_1d sd_stress(d_stress);
  sview_2d sd_forces(d_forces);

  // accumurate forces and stress for each neighbor pair
  Kokkos::parallel_for("forces_and_stress",
                       range_policy(0, n_pairs_),
                       KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx) {
                         const double dis = d_distance_(npidx);
                         if (dis > cutoff_) {
                           return;
                         }
                         const double invdis = 1.0 / dis;

                         const double delx = d_neighbor_pair_displacements(npidx, 0);
                         const double dely = d_neighbor_pair_displacements(npidx, 1);
                         const double delz = d_neighbor_pair_displacements(npidx, 2);
                         const double delx_invdis = delx * invdis;
                         const double dely_invdis = dely * invdis;
                         const double delz_invdis = delz * invdis;

                         const auto &ij = d_neighbor_pair_index(npidx);
                         const LMPLocalIdx i = ij.first;
                         const LMPLocalIdx j = ij.second;

                         // forces acted on i from j minus forces acted on j from i
                         double fx = 0.0;
                         double fy = 0.0;
                         double fz = 0.0;
                         const TypeCombIdx tc_ij = d_neighbor_pair_typecomb(npidx);

                         if (j < nlocal_ && j < i) {
                           for (int n = 0; n < n_fn_; ++n) {
                             for (LMInfoIdx lmi = 0; lmi < n_lm_half_; ++lmi) {
                               const int m = d_lm_info(lmi, 1);
                               const double coeff = (m == 0) ? 1.0 : 2.0;

                               const auto f1 = d_fn_der_(npidx, n) * d_ylm_(npidx, lmi);
                               auto basis_function_dx = f1 * delx_invdis + d_fn_(npidx, n) * d_ylm_dx_(npidx, lmi);
                               auto basis_function_dy = f1 * dely_invdis + d_fn_(npidx, n) * d_ylm_dy_(npidx, lmi);
                               auto basis_function_dz = f1 * delz_invdis + d_fn_(npidx, n) * d_ylm_dz_(npidx, lmi);
                               basis_function_dx.imag() *= -1;
                               basis_function_dy.imag() *= -1;
                               basis_function_dz.imag() *= -1;

                               fx += coeff * product_real_part(d_basis_function_adjoints_(i, tc_ij, n, lmi),
                                                               basis_function_dx);
                               fy += coeff * product_real_part(d_basis_function_adjoints_(i, tc_ij, n, lmi),
                                                               basis_function_dy);
                               fz += coeff * product_real_part(d_basis_function_adjoints_(i, tc_ij, n, lmi),
                                                               basis_function_dz);

                               // sign for parity of spherical harmonics, (-1)^l
                               const int l = d_lm_info(lmi, 0);
                               const double scale = (l % 2) ? -1.0 : 1.0;
                               fx += coeff * scale * product_real_part(d_basis_function_adjoints_(j, tc_ij, n, lmi),
                                                                       basis_function_dx);
                               fy += coeff * scale * product_real_part(d_basis_function_adjoints_(j, tc_ij, n, lmi),
                                                                       basis_function_dy);
                               fz += coeff * scale * product_real_part(d_basis_function_adjoints_(j, tc_ij, n, lmi),
                                                                       basis_function_dz);
                             }
                           }
                         }
                         if (j >= nlocal_) {
                           for (int n = 0; n < n_fn_; ++n) {
                             for (LMInfoIdx lmi = 0; lmi < n_lm_half_; ++lmi) {
                               const int m = d_lm_info(lmi, 1);
                               const double coeff = (m == 0) ? 1.0 : 2.0;

                               const auto f1 = d_fn_der_(npidx, n) * d_ylm_(npidx, lmi);
                               auto basis_function_dx = f1 * delx_invdis + d_fn_(npidx, n) * d_ylm_dx_(npidx, lmi);
                               auto basis_function_dy = f1 * dely_invdis + d_fn_(npidx, n) * d_ylm_dy_(npidx, lmi);
                               auto basis_function_dz = f1 * delz_invdis + d_fn_(npidx, n) * d_ylm_dz_(npidx, lmi);
                               basis_function_dx.imag() *= -1;
                               basis_function_dy.imag() *= -1;
                               basis_function_dz.imag() *= -1;

                               fx += coeff * product_real_part(d_basis_function_adjoints_(i, tc_ij, n, lmi),
                                                               basis_function_dx);
                               fy += coeff * product_real_part(d_basis_function_adjoints_(i, tc_ij, n, lmi),
                                                               basis_function_dy);
                               fz += coeff * product_real_part(d_basis_function_adjoints_(i, tc_ij, n, lmi),
                                                               basis_function_dz);
                             }
                           }
                         }
                         // update forces
                         auto sd_forces_a = sd_forces.access();
                         sd_forces_a(i, 0) += fx;
                         sd_forces_a(i, 1) += fy;
                         sd_forces_a(i, 2) += fz;
                         sd_forces_a(j, 0) -= fx;
                         sd_forces_a(j, 1) -= fy;
                         sd_forces_a(j, 2) -= fz;

                         // update stress tensor
                         auto sd_stress_a = sd_stress.access();
                         sd_stress_a(0) -= d_neighbor_pair_displacements(npidx, 0) * fx; // xx
                         sd_stress_a(1) -= d_neighbor_pair_displacements(npidx, 1) * fy; // yy
                         sd_stress_a(2) -= d_neighbor_pair_displacements(npidx, 2) * fz; // zz
                         sd_stress_a(3) -= d_neighbor_pair_displacements(npidx, 1) * fz; // yz
                         sd_stress_a(4) -= d_neighbor_pair_displacements(npidx, 2) * fx; // zx
                         sd_stress_a(5) -= d_neighbor_pair_displacements(npidx, 0) * fy; // xy
                       }
  );
  Kokkos::Experimental::contribute(d_forces, sd_forces);
  Kokkos::Experimental::contribute(d_stress, sd_stress);
  forces_kk_.modify_device();
  stress_kk_.modify_device();
  Kokkos::fence();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_forces_and_stress(NeighFull neighflag,
                                                                         NewtonOff newton_pair,
                                                                         NeighListKokkos *k_list) {
  ;  // do nothing
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_forces_and_stress(NeighHalfThread neighflag,
                                                                         NewtonOn newton_pair,
                                                                         NeighListKokkos *k_list) {
  MLIPModel::compute_forces_and_stress();
}

template<class PairStyle, class NeighListKokkos>
void MLIPModelLMP<PairStyle, NeighListKokkos>::compute_forces_and_stress(NeighHalf neighflag,
                                                                         NewtonOn newton_pair,
                                                                         NeighListKokkos *k_list) {
  MLIPModel::compute_forces_and_stress();
}

template<class PairStyle, class NeighListKokkos>
vector1d MLIPModelLMP<PairStyle, NeighListKokkos>::get_stress(){
  stress_kk_.sync_host();

  vector1d stress(6);
  const auto h_stress = stress_kk_.view_host();
  stress[0] = h_stress(0);  // xx
  stress[1] = h_stress(1);  // yy
  stress[2] = h_stress(2);  // zz
  stress[3] = h_stress(5);  // xy
  stress[4] = h_stress(4);  // xz
  stress[5] = h_stress(3);  // yz
  return stress;
}

// ----------------------------------------------------------------------------
// Inline functions
// ----------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
double MLIPModel::product_real_part(const Kokkos::complex<double> &lhs, const Kokkos::complex<double> &rhs) const {
  return lhs.real() * rhs.real() - lhs.imag() * rhs.imag();
}

}  // MLIP_NS

#endif //LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_MLIPKK_LMP_H_
