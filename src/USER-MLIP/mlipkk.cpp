#include "mlipkk.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <chrono>
#include <cstring>

#include "mlipkk_types.h"
#include "mlipkk_gtinv_data_reader.h"
#include "mlipkk_irreps_type.h"
#include "mlipkk_polynomial.h"
#include "mlipkk_features.h"
#include "mlipkk_features_kk.h"
#include "mlipkk_types.h"
#include "mlipkk_utils.h"
#include "mlipkk_spherical_harmonics_kk.h"

namespace MLIP_NS {

// ----------------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------------

void MLIPModel::initialize(const MLIPInput& input, const vector1d& reg_coeffs, const Readgtinv& gtinvdata)
{
    n_types_ = input.n_type;
    cutoff_ = input.cutoff;
    model_type_ = input.model_type;
    maxp_ = input.maxp;
    maxl_ = input.maxl;
    pair_type_id_ = input.pair_type_id;

    // regression coefficients
    n_reg_coeffs_ = static_cast<int>(reg_coeffs.size());
    Kokkos::resize(reg_coeffs_kk_, n_reg_coeffs_);  // resize on device
    reg_coeffs_kk_.sync_host();  // sync from device to host and reset modified_flag to 0
    auto h_reg_coeffs = reg_coeffs_kk_.view_host();
    for (int i = 0; i < n_reg_coeffs_; ++i) {
        h_reg_coeffs(i) = reg_coeffs[i];
    }
    reg_coeffs_kk_.modify_host();
    reg_coeffs_kk_.sync_device();

    initialize_radial_basis(pair_type_id_, input.params);  // set n_fn_ here
    initialize_gtinv(n_types_, n_fn_, model_type_, maxp_, gtinvdata);
    initialize_typecomb(n_types_, n_typecomb_);
    initialize_sph(maxl_);
    Kokkos::fence();
}

void MLIPModel::initialize_radial_basis(const int pair_type_id, const vector2d& params) {
    n_fn_ = static_cast<int>(params.size());
    Kokkos::resize(radial_params_kk_, n_fn_, 2);
    radial_params_kk_.sync_host();
    auto h_params = radial_params_kk_.view_host();
    for (int n = 0; n < n_fn_; ++n) {
        h_params(n, 0) = params[n][0];
        h_params(n, 1) = params[n][1];
    }
    radial_params_kk_.modify_host();
    radial_params_kk_.sync_device();

    Kokkos::fence();
}

void MLIPModel::initialize_gtinv(const int n_types, const int n_fn, const int model_type, const int maxp,
                                 const Readgtinv& gtinvdata)
{
    // Load Irreps data
    const auto& l_array = gtinvdata.get_l_comb();
    d_lm_array_ = Kokkos::create_staticcrsgraph<StaticCrsGraph>("d_lm_array_", gtinvdata.get_flatten_lm_seq());

    const auto& flatten_lm_coeffs = gtinvdata.get_flatten_lm_coeffs();
    const int n_irrepsterm = static_cast<int>(flatten_lm_coeffs.size());
    Kokkos::resize(lm_coeffs_kk_, n_irrepsterm);
    lm_coeffs_kk_.sync_host();
    auto h_lm_coeffs = lm_coeffs_kk_.view_host();
    for (IrrepsTermIdx iterm = 0; iterm < n_irrepsterm; ++iterm) {
        h_lm_coeffs(iterm) = flatten_lm_coeffs[iterm];
    }
    lm_coeffs_kk_.modify_host();
    lm_coeffs_kk_.sync_device();

    // construct pairs of Irreps and types
    const auto irreps_type_pairs = get_unique_irreps_type_pairs(n_types, l_array);
    n_irreps_typecomb_ = static_cast<int>(irreps_type_pairs.size());
    n_des_ = n_fn * n_irreps_typecomb_;

    // type combination mapping
    n_typecomb_ = n_types * (n_types + 1) / 2;

    std::vector<std::vector<TypeCombIdx>> irreps_type_combs;
    for (IrrepsTypeCombIdx itcidx = 0; itcidx < n_irreps_typecomb_; ++itcidx) {
        const IrrepsTypePair& itp = irreps_type_pairs[itcidx];
        irreps_type_combs.emplace_back(itp.type_combs);
    }
    d_irreps_type_combs_ = Kokkos::create_staticcrsgraph<StaticCrsGraph>("d_irreps_type_combs_", irreps_type_combs);

    Kokkos::resize(irreps_type_intersection_, n_irreps_typecomb_, n_types_);
    irreps_type_intersection_.sync_host();
    auto h_irreps_type_intersection = irreps_type_intersection_.view_host();
    for (IrrepsTypeCombIdx itcidx = 0; itcidx < n_irreps_typecomb_; ++itcidx) {
        const IrrepsTypePair& itp = irreps_type_pairs[itcidx];
        for (ElementType type = 0; type < n_types_; ++type) {
            h_irreps_type_intersection(itcidx, type) = itp.type_intersection[type];
        }
    }
    irreps_type_intersection_.modify_host();
    irreps_type_intersection_.sync_device();

    Kokkos::resize(irreps_type_mapping_, n_irreps_typecomb_);
    irreps_type_mapping_.sync_host();
    auto h_irreps_type_mapping = irreps_type_mapping_.view_host();
    for (IrrepsTypeCombIdx itcidx = 0; itcidx < n_irreps_typecomb_; ++itcidx) {
        const IrrepsTypePair& itp = irreps_type_pairs[itcidx];
        h_irreps_type_mapping(itcidx) = itp.irreps_idx;
    }
    irreps_type_mapping_.modify_host();
    irreps_type_mapping_.sync_device();

    const auto& irreps_first_term = gtinvdata.get_irreps_first_term();
    const int num_irreps = static_cast<int>(l_array.size());
    Kokkos::resize(irreps_first_term_, num_irreps);
    irreps_first_term_.sync_host();
    auto h_irreps_first_term = irreps_first_term_.view_host();
    for (IrrepsIdx iidx = 0; iidx < num_irreps; ++iidx) {
        h_irreps_first_term(iidx) = irreps_first_term[iidx];
    }
    irreps_first_term_.modify_host();
    irreps_first_term_.sync_device();

    const auto& irreps_num_terms = gtinvdata.get_irreps_num_terms();
    Kokkos::resize(irreps_num_terms_, num_irreps);
    irreps_num_terms_.sync_host();
    auto h_irreps_num_terms = irreps_num_terms_.view_host();
    for (IrrepsIdx iidx = 0; iidx < num_irreps; ++iidx) {
        h_irreps_num_terms(iidx) = irreps_num_terms[iidx];
    }
    irreps_num_terms_.modify_host();
    irreps_num_terms_.sync_device();

    // polynomial index
    MLIPPolynomial poly(model_type, maxp, n_fn, n_types, irreps_type_pairs);
    d_polynomial_index_ = Kokkos::create_staticcrsgraph<StaticCrsGraph>("d_polynomial_index_", poly.get_polynomial_index());

    Kokkos::fence();
}

/// initialize variables related to type combination
void MLIPModel::initialize_typecomb(const int n_types, const int n_typecomb) {
    // the other type of a type combination
    Kokkos::resize(other_type_kk_, n_typecomb, n_types);
    other_type_kk_.sync_host();
    auto h_other_type = other_type_kk_.view_host();
    // if other type does not exist, return -1
    for (TypeCombIdx tcidx = 0; tcidx < n_typecomb; ++tcidx) {
        for (ElementType type = 0; type < n_types; ++type) {
            h_other_type(tcidx, type) = -1;
        }
    }
    const auto type_pairs_mapping = get_type_pairs_mapping(n_types);
    for (TypeCombIdx tcidx = 0; tcidx < n_typecomb; ++tcidx) {
        const auto& p = type_pairs_mapping[tcidx];
        h_other_type(tcidx, p.first) = p.second;
        h_other_type(tcidx, p.second) = p.first;
    }
    other_type_kk_.modify_host();
    other_type_kk_.sync_device();

    // type combinations
    Kokkos::resize(type_pairs_kk_, n_types, n_types);
    type_pairs_kk_.sync_host();
    auto h_type_pairs = type_pairs_kk_.view_host();
    int count = 0;
    for (ElementType type1 = 0; type1 < n_types; ++type1) {
        for (ElementType type2 = type1; type2 < n_types; ++type2) {
            h_type_pairs(type1, type2) = count;
            h_type_pairs(type2, type1) = count;
            ++count;
        }
    }
    type_pairs_kk_.modify_host();
    type_pairs_kk_.sync_device();

    Kokkos::fence();
}

/// intialize variables related to spherical harmonics
void MLIPModel::initialize_sph(const int maxl) {
    // utility for LMIdx
    n_lm_all_ = (maxl_ + 1) * (maxl_ + 1);
    n_lm_half_ = (maxl_ + 1) * (maxl_ + 2) / 2;

    // spherical harmonics
    Kokkos::resize(lm_info_kk_, n_lm_half_, 4);
    lm_info_kk_.sync_host();
    auto h_lm_info = lm_info_kk_.view_host();
    for (int l = 0; l <= maxl_; ++l){
        for (int m = -l; m <= 0; ++m){
            const LMInfoIdx lmi = l * (l + 1) / 2 + l + m;
            const LMIdx lm1 = l * l + l + m;
            const LMIdx lm2 = l * l + l - m;
            h_lm_info(lmi, 0) = l;
            h_lm_info(lmi, 1) = m;
            h_lm_info(lmi, 2) = lm1;
            h_lm_info(lmi, 3) = lm2;

        }
    }
    lm_info_kk_.modify_host();
    lm_info_kk_.sync_device();

    // utility for converting LMIdx to l and m
    Kokkos::resize(lm2l_, n_lm_all_);
    lm2l_.sync_host();
    Kokkos::resize(lm2m_, n_lm_all_);
    lm2m_.sync_host();
    auto h_lm2l = lm2l_.view_host();
    auto h_lm2m = lm2m_.view_host();
    for (int l = 0; l <= maxl_; ++l) {
        for (int m = -l; m <= l; ++m) {
            LMIdx lm = l * l + l + m;
            h_lm2l(lm) = l;
            h_lm2m(lm) = m;
        }
    }
    lm2l_.modify_host();
    lm2m_.modify_host();
    lm2l_.sync_device();
    lm2m_.sync_device();

    // precompute coefficients to compute spherical harmonics
    initialize_sph_AB(maxl);

    Kokkos::fence();
}

void MLIPModel::initialize_sph_AB(const int maxl) {
    Kokkos::resize(sph_coeffs_A_, n_lm_half_);
    sph_coeffs_A_.sync_host();
    Kokkos::resize(sph_coeffs_B_, n_lm_half_);
    sph_coeffs_B_.sync_host();

    auto h_sph_coeffs_A = sph_coeffs_A_.view_host();
    auto h_sph_coeffs_B = sph_coeffs_B_.view_host();

    for (int l = 2; l <= maxl; ++l) {
        double ls = l * l;
        double lm1s = (l - 1) * (l - 1);
        for (int m = 0; m <= l - 2; ++m) {
            double ms = m * m;
            h_sph_coeffs_A(lm2i(l, m)) = sqrt((4.0 * ls - 1.0) / (ls - ms));
            h_sph_coeffs_B(lm2i(l, m)) = -sqrt((lm1s - ms) / (4.0 * lm1s - 1.0));
        }
    }
    sph_coeffs_A_.modify_host();
    sph_coeffs_B_.modify_host();
    sph_coeffs_A_.sync_device();
    sph_coeffs_B_.sync_device();

    Kokkos::fence();
}

// ----------------------------------------------------------------------------
// Preparation for coming structure
// ----------------------------------------------------------------------------

void MLIPModel::set_structure(const std::vector<ElementType>& types,
                              const vector3d& displacements, const std::vector<std::vector<SiteIdx>>& neighbors)
{
    assert(neighbors.size() == types.size());

    inum_ = static_cast<int>(displacements.size());

    // number of (i, j) neighbors
    // TODO: atom-first indexing for GPU
    n_pairs_ = 0;
    for (SiteIdx i = 0; i < inum_; ++i) {
        n_pairs_ += static_cast<int>(neighbors[i].size());
    }

    // flatten half-neighbor list
    // neighbor_pair_index and neighbor_pair_displacements
    Kokkos::resize(neighbor_pair_index_kk_, n_pairs_);
    neighbor_pair_index_kk_.sync_host();
    Kokkos::resize(neighbor_pair_displacements_kk_, n_pairs_, 3);
    neighbor_pair_displacements_kk_.sync_host();
    auto h_neighbor_pair_index = neighbor_pair_index_kk_.view_host();
    auto h_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_host();

    NeighborPairIdx count_neighbor = 0;
    for (SiteIdx i = 0; i < inum_; ++i) {
        const int num_neighbors_i = static_cast<int>(neighbors[i].size());
        for (int jj = 0; jj < num_neighbors_i; ++jj) {
            const SiteIdx j = neighbors[i][jj];
            h_neighbor_pair_index(count_neighbor) = Kokkos::pair<SiteIdx, SiteIdx>(i, j);
            for (int x = 0; x < 3; ++x) {
                h_neighbor_pair_displacements(count_neighbor, x) = displacements[i][jj][x];
            }
            ++count_neighbor;
        }
    }
    neighbor_pair_index_kk_.modify_host();
    neighbor_pair_displacements_kk_.modify_host();
    neighbor_pair_index_kk_.sync_device();
    neighbor_pair_displacements_kk_.sync_device();

    // neighbor_pair_typecomb
    Kokkos::resize(neighbor_pair_typecomb_kk_, n_pairs_);
    neighbor_pair_typecomb_kk_.sync_host();
    auto h_neighbor_pair_typecomb = neighbor_pair_typecomb_kk_.view_host();
    for (NeighborPairIdx npidx = 0; npidx < n_pairs_; ++npidx) {
        const auto& ij = h_neighbor_pair_index(npidx);
        const SiteIdx i = ij.first;
        const SiteIdx j = ij.second;
        const ElementType type_i = types[i];
        const ElementType type_j = types[j];
        h_neighbor_pair_typecomb(npidx) = type_pairs_kk_.h_view(type_i, type_j);
    }
    neighbor_pair_typecomb_kk_.modify_host();
    neighbor_pair_typecomb_kk_.sync_device();

    // types
    Kokkos::resize(types_kk_, inum_);
    types_kk_.sync_host();
    auto h_types = types_kk_.view_host();
    for (SiteIdx i = 0; i < inum_; ++i) {
        h_types(i) = types[i];
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
    structural_features_kk_.sync_host();
    Kokkos::resize(d_polynomial_adjoints_, inum_, n_des_);
    Kokkos::resize(d_basis_function_adjoints_, inum_, n_typecomb_, n_fn_, n_lm_half_);

    Kokkos::resize(site_energy_kk_, inum_);
    site_energy_kk_.sync_host();
    Kokkos::resize(forces_kk_, inum_, 3);
    forces_kk_.sync_host();

    Kokkos::fence();
}

// ----------------------------------------------------------------------------
// Compute energy, forces, and stress
// ----------------------------------------------------------------------------

void MLIPModel::compute() {
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

    compute_order_parameters();

#ifdef _DEBUG
    time_op += std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now() - end).count();
    std::cerr << "anlm     : " << time_op << " ns" << std::endl;
    end = std::chrono::system_clock::now();
#endif

    compute_structural_features();

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

void MLIPModel::prepare_features() {
    compute_basis_functions();
    compute_order_parameters();
    compute_structural_features();

    // the remained functions are only needed for fitting
    compute_polynomial_features();
    compute_dirj();
    compute_force_and_stress_features();
}

/// update reg_coeffs for training
void MLIPModel::update_reg_coeffs(const vector1d& reg_coeffs) {
    if (static_cast<int>(reg_coeffs.size()) != n_reg_coeffs_) {
        std::cerr << "Invalid size for regression coefficients: "
                  << static_cast<int>(reg_coeffs.size()) << std::endl;
        exit(1);
    }
    auto h_reg_coeffs = reg_coeffs_kk_.view_host();
    for (int i = 0; i < n_reg_coeffs_; ++i) {
        h_reg_coeffs(i) = reg_coeffs[i];
    }
    reg_coeffs_kk_.modify_host();
    reg_coeffs_kk_.sync_device();
    Kokkos::fence();
}

void MLIPModel::compute_basis_functions() {
    const auto d_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_device();
    const auto d_sph_coeffs_A = sph_coeffs_A_.view_device();
    const auto d_sph_coeffs_B = sph_coeffs_B_.view_device();

    Kokkos::parallel_for("ylm",
        range_policy(0, n_pairs_),
        KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx) {
            const double delx = d_neighbor_pair_displacements(npidx, 0);
            const double dely = d_neighbor_pair_displacements(npidx, 1);
            const double delz = d_neighbor_pair_displacements(npidx, 2);

            const double r = sqrt(delx * delx + dely * dely + delz * delz);
            d_distance_(npidx) = r;
            if (r > cutoff_) {
                return;
            }

            const double costheta = delz / r;
            const double azimuthal = atan2(dely, delx);

            // spherical harmonics and derivative of spherical harmonics in cartesian coords.
            compute_alp(npidx, costheta, maxl_, d_sph_coeffs_A, d_sph_coeffs_B, d_alp_);
            compute_ylm(npidx, azimuthal, maxl_, d_alp_, d_ylm_);
            compute_alp_sintheta(npidx, costheta, maxl_, d_sph_coeffs_A, d_sph_coeffs_B, d_alp_sintheta_);
            compute_ylm_der(npidx, costheta, azimuthal, r, maxl_, d_alp_sintheta_,
                            d_ylm_dx_, d_ylm_dy_, d_ylm_dz_);
        }
    );

    const auto d_params = radial_params_kk_.view_device();

    Kokkos::parallel_for("fn",
        range_policy(0, n_pairs_),
        KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx) {
            const double r = d_distance_(npidx);
            if (r > cutoff_) {
                return;
            }

            // radial functions
            get_fn_kk(npidx, r, cutoff_, d_params, d_fn_, d_fn_der_);
        }
    );

    Kokkos::fence();
}

void MLIPModel::compute_order_parameters() {
    Kokkos::parallel_for("init_anlm_half",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {inum_, n_types_, n_fn_, n_lm_half_}),
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
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>({0, 0, 0, 0}, {inum_, n_types_, n_fn_, n_lm_half_}),
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

void MLIPModel::compute_structural_features() {
    const auto d_types = types_kk_.view_device();
    const auto d_other_type = other_type_kk_.view_device();
    const auto d_irreps_type_intersection = irreps_type_intersection_.view_device();
    const auto d_irreps_type_mapping = irreps_type_mapping_.view_device();
    const auto d_irreps_first_term = irreps_first_term_.view_device();
    const auto d_irreps_num_terms = irreps_num_terms_.view_device();
    const auto d_lm_coeffs = lm_coeffs_kk_.view_device();
    auto d_structural_features = structural_features_kk_.view_device();

    Kokkos::parallel_for("init_structural_features",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, n_des_}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const FeatureIdx fidx) {
            d_structural_features(i, fidx) = 0.0;
        }
    );

    Kokkos::parallel_for("structural_features",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, n_des_}),
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

void MLIPModel::compute_polynomial_adjoints() {
    const int num_poly_idx = n_reg_coeffs_;

    const auto d_reg_coeffs = reg_coeffs_kk_.view_device();
    const auto d_structural_features = structural_features_kk_.view_device();
    sview_2d sd_polynomial_adjoints(d_polynomial_adjoints_);

    Kokkos::parallel_for("init_polynomial_adjoints",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, n_des_}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const FeatureIdx fidx) {
            d_polynomial_adjoints_(i, fidx) = 0.0;
        }
    );

    // scratch memory for d_structural_features(i, :)
    using ScratchPadView = Kokkos::View<double*, ExecSpace::scratch_memory_space>;
    size_t scratch_bytes = ScratchPadView::shmem_size(n_des_);
    const int scratch_level = 0;

    Kokkos::parallel_for("polynomial_adjoints",
        team_policy(inum_, Kokkos::AUTO).set_scratch_size(scratch_level, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_CLASS_LAMBDA(const team_policy::member_type& teamMember) {
            const SiteIdx i = teamMember.league_rank();

            // load into scratch
            const ScratchPadView d_structural_features_i(teamMember.team_scratch(scratch_level), n_des_);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(teamMember, n_des_),
                [=, *this] (const FeatureIdx fidx) {
                    d_structural_features_i(fidx) = d_structural_features(i, fidx);
                }
            );
            teamMember.team_barrier();

            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(teamMember, num_poly_idx),
                [=, *this] (const PolynomialIdx pidx) {
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

void MLIPModel::compute_basis_function_adjoints() {
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

    Kokkos::parallel_for("init_basis_function_adjoints",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<4>>({0, 0, 0, 0}, {inum_, n_typecomb_, n_fn_, n_lm_half_}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const TypeCombIdx tcidx, const int n, const LMInfoIdx lmi) {
            d_basis_function_adjoints_(i, tcidx, n, lmi) = 0.0;
        }
    );

    Kokkos::parallel_for("basis_function_adjoints",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {inum_, n_irreps_typecomb_, n_fn_}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const IrrepsTypeCombIdx itcidx, const int n) {
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

            const FeatureIdx fidx = n * n_irreps_typecomb_ + itcidx;  // consistent with poly_.get_feature_idx
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

void MLIPModel::compute_energy() {
    const int num_poly_idx = n_reg_coeffs_;
    auto d_site_energy = site_energy_kk_.view_device();

    // initialize
    Kokkos::parallel_for("init_energy",
        range_policy(0, inum_),
        KOKKOS_LAMBDA(const SiteIdx i) {
            d_site_energy(i) = 0.0;
        }
    );

    const auto d_reg_coeffs = reg_coeffs_kk_.view_device();
    const auto d_structural_features = structural_features_kk_.view_device();

    // energy for each atom-i
    sview_1d sd_site_energy(d_site_energy);
    Kokkos::parallel_for("site_energy",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, num_poly_idx}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const PolynomialIdx pidx) {
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
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, double& energy_tmp) {
            energy_tmp += site_energy_kk_.d_view(i);
        },
        energy
    );
    Kokkos::fence();
    energy_ = energy;

    Kokkos::fence();
}

void MLIPModel::compute_forces_and_stress() {
    auto d_forces = forces_kk_.view_device();
    auto d_stress = stress_kk_.view_device();

    // initialize
    Kokkos::parallel_for("init_forces",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {inum_, 3}),
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

            const auto& ij = d_neighbor_pair_index(npidx);
            const SiteIdx i = ij.first;
            const SiteIdx j = ij.second;

            // forces acted on i from j minus forces acted on j from i
            double fx = 0.0;
            double fy = 0.0;
            double fz = 0.0;
            const TypeCombIdx tc_ij = d_neighbor_pair_typecomb(npidx);

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

void MLIPModel::compute_dirj() {
    Kokkos::resize(d_dirj_, n_pairs_, n_des_, 3);
    Kokkos::resize(d_djri_, n_pairs_, n_des_, 3);

    // init
    Kokkos::parallel_for("init_dirj",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n_pairs_, n_des_}),
        KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx, const FeatureIdx fidx) {
            for (int x = 0; x < 3; ++x) {
                d_dirj_(npidx, fidx, x) = 0.0;
                d_djri_(npidx, fidx, x) = 0.0;
            }
        }
    );

    const auto d_lm2l = lm2l_.view_device();
    const auto d_lm2m = lm2m_.view_device();
    const auto d_types = types_kk_.view_device();
    const auto d_other_type = other_type_kk_.view_device();
    const auto d_neighbor_pair_index = neighbor_pair_index_kk_.view_device();
    const auto d_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_device();
    const auto d_neighbor_pair_typecomb = neighbor_pair_typecomb_kk_.view_device();
    const auto d_irreps_type_mapping = irreps_type_mapping_.view_device();
    const auto d_irreps_first_term = irreps_first_term_.view_device();
    const auto d_irreps_num_terms = irreps_num_terms_.view_device();
    const auto d_lm_coeffs = lm_coeffs_kk_.view_device();

    Kokkos::parallel_for("dirj",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n_pairs_, n_des_}),
        KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx, const FeatureIdx fidx) {
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

            const auto& ij = d_neighbor_pair_index(npidx);
            const SiteIdx i = ij.first;
            const SiteIdx j = ij.second;
            const TypeCombIdx tc_ij = d_neighbor_pair_typecomb(npidx);
            const ElementType type_i = d_types(i);
            const ElementType type_j = d_types(j);

            // should be consistent with poly_.get_irreps_type_idx
            const IrrepsTypeCombIdx itcidx = fidx % n_irreps_typecomb_;
            auto type_combs_rowview = d_irreps_type_combs_.rowConst(itcidx);
            const IrrepsIdx iidx = d_irreps_type_mapping(itcidx);
            const IrrepsTermIdx first_term = d_irreps_first_term(iidx);
            const int num_terms = d_irreps_num_terms(iidx);
            const int order = type_combs_rowview.length;

            const int n = fidx / n_irreps_typecomb_;

            for (int term = 0; term < num_terms; ++term) {
                const IrrepsTermIdx iterm = first_term + term;
                auto lm_term = d_lm_array_.rowConst(iterm);

                for (int mu = 0; mu < order; ++mu) {
                    const TypeCombIdx tcidx_mu = type_combs_rowview(mu);
                    if (tcidx_mu != tc_ij) {
                        continue;
                    }

                    Kokkos::complex<double> tmp_i = d_lm_coeffs(iterm);
                    Kokkos::complex<double> tmp_j = d_lm_coeffs(iterm);
                    for (int mu2 = 0; mu2 < order; ++mu2) {
                        if (mu2 == mu) {
                            continue;
                        }
                        const TypeCombIdx tcidx_mu2 = type_combs_rowview(mu2);
                        const LMIdx lm_mu2 = lm_term(mu2);

                        const ElementType type_i_mu2 = d_other_type(tcidx_mu2, type_i);
                        if (type_i_mu2 != -1) {
                            tmp_i *= d_anlm_(i, type_i_mu2, n, lm_mu2);
                        } else {
                            tmp_i = 0.0;
                        }

                        const ElementType type_j_mu2 = d_other_type(tcidx_mu2, type_j);
                        if (type_j_mu2 != -1) {
                            tmp_j *= d_anlm_(j, type_j_mu2, n, lm_mu2);
                        } else {
                            tmp_j = 0.0;
                        }

                    }

                    const LMIdx lm_mu = lm_term(mu);
                    const int l_mu = d_lm2l(lm_mu);
                    const int m_mu = d_lm2m(lm_mu);
                    Kokkos::complex<double> ylm, ylm_dx, ylm_dy, ylm_dz;
                    if (m_mu > 0) {
                        const LMInfoIdx lmi_mu = l_mu * (l_mu + 1) / 2 + l_mu - m_mu;  // (l_mu, -m_mu)
                        const double scale_m = (m_mu % 2) ? -1.0 : 1.0;

                        ylm = d_ylm_(npidx, lmi_mu);
                        ylm_dx = d_ylm_dx_(npidx, lmi_mu);
                        ylm_dy = d_ylm_dy_(npidx, lmi_mu);
                        ylm_dz = d_ylm_dz_(npidx, lmi_mu);

                        ylm = Kokkos::complex<double>(scale_m * ylm.real(), -scale_m * ylm.imag());
                        ylm_dx = Kokkos::complex<double>(scale_m * ylm_dx.real(), -scale_m * ylm_dx.imag());
                        ylm_dy = Kokkos::complex<double>(scale_m * ylm_dy.real(), -scale_m * ylm_dy.imag());
                        ylm_dz = Kokkos::complex<double>(scale_m * ylm_dz.real(), -scale_m * ylm_dz.imag());
                    } else {
                        const LMInfoIdx lmi_mu = l_mu * (l_mu + 1) / 2 + l_mu + m_mu;  // (l_mu, m_mu)
                        ylm = d_ylm_(npidx, lmi_mu);
                        ylm_dx = d_ylm_dx_(npidx, lmi_mu);
                        ylm_dy = d_ylm_dy_(npidx, lmi_mu);
                        ylm_dz = d_ylm_dz_(npidx, lmi_mu);
                    }

                    const auto f1 = d_fn_der_(npidx, n) * ylm;
                    auto basis_function_dx = f1 * delx_invdis + d_fn_(npidx, n) * ylm_dx;
                    auto basis_function_dy = f1 * dely_invdis + d_fn_(npidx, n) * ylm_dy;
                    auto basis_function_dz = f1 * delz_invdis + d_fn_(npidx, n) * ylm_dz;
                    basis_function_dx.imag() *= -1;
                    basis_function_dy.imag() *= -1;
                    basis_function_dz.imag() *= -1;

                    d_dirj_(npidx, fidx, 0) += (tmp_i * basis_function_dx).real();
                    d_dirj_(npidx, fidx, 1) += (tmp_i * basis_function_dy).real();
                    d_dirj_(npidx, fidx, 2) += (tmp_i * basis_function_dz).real();

                    // sign for parity of spherical harmonics, (-1)^l
                    const double scale = (l_mu % 2) ? -1.0 : 1.0;
                    d_djri_(npidx, fidx, 0) -= (scale * tmp_j * basis_function_dx).real();
                    d_djri_(npidx, fidx, 1) -= (scale * tmp_j * basis_function_dy).real();
                    d_djri_(npidx, fidx, 2) -= (scale * tmp_j * basis_function_dz).real();

                }
            }
        }
    );

    Kokkos::fence();
}

void MLIPModel::compute_polynomial_features() {
    const auto d_structural_features = structural_features_kk_.view_device();

    const int num_poly_idx = n_reg_coeffs_;

    Kokkos::resize(polynomial_features_kk_, inum_, num_poly_idx);
    auto d_polynomial_features = polynomial_features_kk_.view_device();

    Kokkos::parallel_for("polynomail_features",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {inum_, num_poly_idx}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const PolynomialIdx pidx) {
            double feature = 1.0;
            auto rowview = d_polynomial_index_.rowConst(pidx);
            for (int p = 0; p < rowview.length; ++p) {
                const FeatureIdx fidx = rowview(p);
                feature *= d_structural_features(i, fidx);
            }
            d_polynomial_features(i, pidx) = feature;
        }
    );

    polynomial_features_kk_.modify_device();
    Kokkos::fence();
}

void MLIPModel::compute_force_and_stress_features() {
    const auto d_neighbor_pair_index = neighbor_pair_index_kk_.view_device();

    const auto d_structural_features = structural_features_kk_.view_device();

    Kokkos::resize(force_features_kk_, inum_, n_reg_coeffs_, 3);
    Kokkos::resize(stress_features_kk_, n_reg_coeffs_, 6);
    auto d_force_features = force_features_kk_.view_device();
    auto d_stress_features = stress_features_kk_.view_device();

    Kokkos::parallel_for("init_force_features",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<3>>({0, 0, 0}, {inum_, n_reg_coeffs_, 3}),
        KOKKOS_CLASS_LAMBDA(const SiteIdx i, const PolynomialIdx pidx, const int x) {
            d_force_features(i, pidx, x) = 0.0;
        }
    );

    Kokkos::parallel_for("init_stress_features",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n_reg_coeffs_, 6}),
        KOKKOS_CLASS_LAMBDA(const PolynomialIdx pidx, const int v) {
            d_stress_features(pidx, v) = 0.0;
        }
    );

    const auto d_neighbor_pair_displacements = neighbor_pair_displacements_kk_.view_device();
    sview_3d s_force_features(d_force_features);
    sview_2d s_stress_features(d_stress_features);

    Kokkos::parallel_for("force_and_stress_features",
        Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<2>>({0, 0}, {n_pairs_, n_reg_coeffs_}),
        KOKKOS_CLASS_LAMBDA(const NeighborPairIdx npidx, const PolynomialIdx pidx) {
            const auto& ij = d_neighbor_pair_index(npidx);
            const SiteIdx i = ij.first;
            const SiteIdx j = ij.second;
            auto rowview = d_polynomial_index_.rowConst(pidx);
            const int polyorder = rowview.length;

            auto s_force_features_a = s_force_features.access();
            auto s_stress_features_a = s_stress_features.access();

            for (int nu = 0; nu < polyorder; ++nu) {
                double prod_i = 1.0;
                double prod_j = 1.0;
                for (int nu2 = 0; nu2 < polyorder; ++nu2) {
                    if (nu2 == nu) {
                        continue;
                    }
                    const FeatureIdx fidx_nu2 = rowview(nu2);
                    prod_i *= d_structural_features(i, fidx_nu2);
                    prod_j *= d_structural_features(j, fidx_nu2);
                }

                const FeatureIdx fidx_nu = rowview(nu);

                const double term_x = d_dirj_(npidx, fidx_nu, 0) * prod_i - d_djri_(npidx, fidx_nu, 0) * prod_j;
                const double term_y = d_dirj_(npidx, fidx_nu, 1) * prod_i - d_djri_(npidx, fidx_nu, 1) * prod_j;
                const double term_z = d_dirj_(npidx, fidx_nu, 2) * prod_i - d_djri_(npidx, fidx_nu, 2) * prod_j;

                s_force_features_a(i, pidx, 0) += term_x;
                s_force_features_a(i, pidx, 1) += term_y;
                s_force_features_a(i, pidx, 2) += term_z;
                s_force_features_a(j, pidx, 0) -= term_x;
                s_force_features_a(j, pidx, 1) -= term_y;
                s_force_features_a(j, pidx, 2) -= term_z;

                s_stress_features_a(pidx, 0) -= d_neighbor_pair_displacements(npidx, 0) * term_x;
                s_stress_features_a(pidx, 1) -= d_neighbor_pair_displacements(npidx, 1) * term_y;
                s_stress_features_a(pidx, 2) -= d_neighbor_pair_displacements(npidx, 2) * term_z;
                s_stress_features_a(pidx, 3) -= d_neighbor_pair_displacements(npidx, 1) * term_z;
                s_stress_features_a(pidx, 4) -= d_neighbor_pair_displacements(npidx, 2) * term_x;
                s_stress_features_a(pidx, 5) -= d_neighbor_pair_displacements(npidx, 0) * term_y;
            }
        }
    );
    Kokkos::Experimental::contribute(d_force_features, s_force_features);
    Kokkos::Experimental::contribute(d_stress_features, s_stress_features);

    force_features_kk_.modify_device();
    stress_features_kk_.modify_device();

    Kokkos::fence();
}

// ----------------------------------------------------------------------------
// Sync results to host
// ----------------------------------------------------------------------------

vector1d MLIPModel::get_site_energies() {
    site_energy_kk_.sync_host();

    vector1d site_energy(inum_, 0.0);
    const auto h_site_energy = site_energy_kk_.view_host();
    for (SiteIdx i = 0; i < inum_; ++i) {
        site_energy[i] = h_site_energy(i);
    }
    return site_energy;
}

vector2d MLIPModel::get_forces() {
    forces_kk_.sync_host();

    vector2d forces(inum_, vector1d(3));
    const auto h_forces = forces_kk_.view_host();
    for (SiteIdx i = 0; i < inum_; ++i) {
        for (int x = 0; x < 3; ++x) {
            forces[i][x] = h_forces(i, x);
        }
    }
    return forces;
}

vector1d MLIPModel::get_stress() {
    stress_kk_.sync_host();

    vector1d stress(6);
    const auto h_stress = stress_kk_.view_host();
    for (int vi = 0; vi < 6; ++vi) {
        stress[vi] = h_stress(vi);
    }
    return stress;
}

vector2d MLIPModel::get_structural_features() {
    structural_features_kk_.sync_host();

    const auto h_structural_features = structural_features_kk_.view_host();
    vector2d structural_features(inum_, vector1d(n_des_));
    for (SiteIdx i = 0; i < inum_; ++i) {
        for (IrrepsTypeCombIdx itcidx = 0; itcidx < n_des_; ++itcidx) {
            structural_features[i][itcidx] = h_structural_features(i, itcidx);
        }
    }
    return structural_features;
}

vector2d MLIPModel::get_polynomial_features() {
    polynomial_features_kk_.sync_host();
    const auto h_polynomial_features = polynomial_features_kk_.view_host();

    const int num_poly_idx = n_reg_coeffs_;
    vector2d polynomial_features(inum_, vector1d(num_poly_idx));

    for (SiteIdx i = 0; i < inum_; ++i) {
        for (PolynomialIdx pidx = 0; pidx < num_poly_idx; ++pidx) {
            polynomial_features[i][pidx] = h_polynomial_features(i, pidx);
        }
    }

    return polynomial_features;
}

vector3d MLIPModel::get_force_features() {
    force_features_kk_.sync_host();
    const auto h_force_features = force_features_kk_.view_host();

    const int num_poly_idx = n_reg_coeffs_;
    vector3d force_features(inum_, vector2d(num_poly_idx, vector1d(3, 0.0)));

    for (SiteIdx i = 0; i < inum_; ++i) {
        for (PolynomialIdx pidx = 0; pidx < num_poly_idx; ++pidx) {
            for (int x = 0; x < 3; ++x) {
                force_features[i][pidx][x] = h_force_features(i, pidx, x);
            }
        }
    }

    return force_features;
}

vector2d MLIPModel::get_stress_features() {
    stress_features_kk_.sync_host();
    const auto h_stress_features = stress_features_kk_.view_host();

    vector2d stress_features(n_reg_coeffs_, vector1d(6));
    for (PolynomialIdx pidx = 0; pidx < n_reg_coeffs_; ++pidx) {
        for (int v = 0; v < 6; ++v) {
            stress_features[pidx][v] = h_stress_features(pidx, v);
        }
    }

    return stress_features;
}

vector1d MLIPModel::get_reg_coeffs() const {
    const auto h_reg_coeffs = reg_coeffs_kk_.view_host();
    vector1d reg_coeffs(n_reg_coeffs_);
    for (PolynomialIdx pidx = 0; pidx < n_reg_coeffs_; ++pidx) {
        reg_coeffs[pidx] = h_reg_coeffs(pidx);
    }
    return reg_coeffs;
}

// ----------------------------------------------------------------------------
// Utilities
// ----------------------------------------------------------------------------

void MLIPModel::dump(std::ostream& os) const {
    int num_structural_features = n_irreps_typecomb_ * n_fn_;
    int num_polynomial_features = n_reg_coeffs_;
    os << "Number of IrrepsTypeComb: "
       << n_irreps_typecomb_ << std::endl;
    os << "Number of StructuralFeature: "
       << num_structural_features
       << " (" << (num_structural_features * 8 / 1024) << " KiB/atom)"
       << std::endl;
    os << "Number of PolynomialFeature: "
       << num_polynomial_features
       << " (" << (num_polynomial_features * 8 / 1024) << " KiB/atom)"
       << std::endl;
    os << "Number of coefficients: "
       << n_reg_coeffs_
       << " (" << (n_reg_coeffs_ * 8 / 1024) << " KiB)"
       << std::endl;
    os << std::endl;

    int required_memory_byte_site = 8 * (n_types_ * n_fn_ * n_lm_half_ * 2 + 2 * n_des_ + 1 + 3)
                                    + 16 * (n_types_ * n_fn_ * n_lm_all_ + n_typecomb_ * n_fn_ * n_lm_half_);
    os << "Required: "
       << (static_cast<double>(required_memory_byte_site) / 1024) << " KiB/atom"
       << std::endl;

    int required_memory_byte_pairs = 4 * (1 + 1)
                                    + 8 * (3 + 1 + 2 * n_fn_ + 2) + 16 * (n_lm_half_ * 4);
    os << "Required: "
       << (static_cast<double>(required_memory_byte_pairs) / 1024) << " KiB/pair"
       << std::endl;
    os << std::endl;

}

// ----------------------------------------------------------------------------
// Inline functions
// ----------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
double MLIPModel::product_real_part(const Kokkos::complex<double>& lhs, const Kokkos::complex<double>& rhs) const {
    return lhs.real() * rhs.real() - lhs.imag() * rhs.imag();
}

} // namespace MLIP
