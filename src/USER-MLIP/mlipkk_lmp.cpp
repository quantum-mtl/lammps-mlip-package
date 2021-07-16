//
// Created on 2021/07/16.
//
#include "kokkos_type.h"
#include "mlipkk_lmp.h"

namespace MLIP_NS{

void MLIPModelLMP::initialize(const MLIPInput& input, const vector1d& reg_coeffs, const Readgtinv& gtinvdata)
{
  MLIPModel::initialize(input, reg_coeffs, gtinvdata);
  nall_ = 0;
}

void MLIPModelLMP::compute() {
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

  MLIPModelLMP::compute_order_parameters();

#ifdef _DEBUG
  time_op += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "anlm     : " << time_op << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  MLIPModelLMP::compute_structural_features();

#ifdef _DEBUG
  time_sf += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "features : " << time_sf << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  compute_energy();

#ifdef _DEBUG
  time_energy += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Energy   : " << time_energy << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  compute_polynomial_adjoints();

#ifdef _DEBUG
  time_pa += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Poly Ad  : " << time_pa << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  compute_basis_function_adjoints();

#ifdef _DEBUG
  time_ba += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Basis Ad : " << time_ba << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

  compute_forces_and_stress();

#ifdef _DEBUG
  time_acc += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "Acc      : " << time_acc << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

}

void MLIPModelLMP::compute_order_parameters() {
  Kokkos::parallel_for("init_anlm_half",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {nall_, n_types_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const SiteIdx i, const ElementType type, const int n, const LMInfoIdx lmi) {
                         d_anlm_r_(i, type, n, lmi) = 0.0;
                         d_anlm_i_(i, type, n, lmi) = 0.0;
                       }
  );

  const auto d_types = types_kk_.view_device();
  const auto d_neighbor_pair_index = neighbor_pair_index_kk_.view_device();
  const auto d_lm_info = lm_info_kk_.view_device();
  sview_4d s_anlm_r (d_anlm_r_);
  sview_4d s_anlm_i (d_anlm_i_);

  // compute order paramters for m <= 0
  Kokkos::parallel_for("anlm_half",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {n_pairs_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx, const int n, const LMInfoIdx lmi) {
                         const auto& ij = d_neighbor_pair_index(npidx);
                         const SiteIdx i = ij.first;
                         const SiteIdx j = ij.second;

                         const ElementType type_i = d_types(i);
                         const ElementType type_j = d_types(j);

                         auto s_anlm_r_a = s_anlm_r.access();
                         auto s_anlm_i_a = s_anlm_i.access();

                         const int l = d_lm_info(lmi, 0);
                         const double scale = (l % 2) ? -1.0 : 1.0;  // sign for parity
                         const Kokkos::complex<double> val = d_fn_(npidx, n) * d_ylm_(npidx, lmi);
                         // neighbors_ is a half list!!!
                         s_anlm_r_a(i, type_j, n, lmi) += val.real();
                         s_anlm_i_a(i, type_j, n, lmi) -= val.imag();  // take c.c.
                         if (i != j) {
                           s_anlm_r_a(j, type_i, n, lmi) += val.real() * scale;
                           s_anlm_i_a(j, type_i, n, lmi) -= val.imag() * scale;  // take c.c
                         }
                       }
  );
  Kokkos::Experimental::contribute(d_anlm_r_, s_anlm_r);
  Kokkos::Experimental::contribute(d_anlm_i_, s_anlm_i);

  // augment order paramters for m > 0
  Kokkos::parallel_for("anlm_all",
                       Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {nall_, n_types_, n_fn_, n_lm_half_}),
                       KOKKOS_CLASS_LAMBDA(const SiteIdx i, const ElementType type, const int n, const LMInfoIdx lmi) {
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

void MLIPModelLMP::compute_structural_features() {
  const auto d_types = types_kk_.view_device();
  const auto d_other_type = other_type_kk_.view_device();
  const auto d_irreps_type_intersection = irreps_type_intersection_.view_device();
  const auto d_irreps_type_mapping = irreps_type_mapping_.view_device();
  const auto d_irreps_first_term = irreps_first_term_.view_device();
  const auto d_irreps_num_terms = irreps_num_terms_.view_device();
  const auto d_lm_coeffs = lm_coeffs_kk_.view_device();
  auto d_structural_features = structural_features_kk_.view_device();

  Kokkos::parallel_for("init_structural_features",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nall_, n_des_}),
                       KOKKOS_CLASS_LAMBDA(const SiteIdx i, const FeatureIdx fidx) {
                         d_structural_features(i, fidx) = 0.0;
                       }
  );

  Kokkos::parallel_for("structural_features",
                       Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {nall_, n_des_}),
                       KOKKOS_CLASS_LAMBDA(const SiteIdx i, const FeatureIdx fidx) {
                         const IrrepsTypeCombIdx itcidx = fidx % n_irreps_typecomb_;  // should be consistent with poly_.get_irreps_type_idx

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
} // namespace MLIP_NS
