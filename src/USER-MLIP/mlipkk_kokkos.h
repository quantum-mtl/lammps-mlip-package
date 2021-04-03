#ifndef MLIP_H
#define MLIP

#include <iostream>
#include <vector>

#include "mlipkk_types.h"
#include "mlipkk_irreps_type.h"
#include "mlipkk_polynomial.h"
#include "mlipkk_features.h"
#include "mlipkk_spherical_harmonics.h"
#include "mlipkk_gtinv_data_reader.h"
#include "mlipkk_potential_parser.h"
#include "mlipkk_types_kokkos.h"

namespace MLIP_NS {

class MLIPModel {
    // ------------------------------------------------------------------------
    // Input parameters
    // ------------------------------------------------------------------------
    int n_types_;
    double cutoff_;
    // 0 for "gaussian"
    int pair_type_id_;
    int model_type_;
    int maxp_;
    int maxl_;
    dview_2d radial_params_kk_ = dview_2d("radial_params_kk_", 0, 0);

    /* number of parameters */
    int n_reg_coeffs_;
    /* number of radial function */
    int n_fn_;
    /* number of type pair combinations */
    int n_typecomb_;
    /* number of IrrepsTypeComb */
    int n_irreps_typecomb_;
    /* number of structural features */
    int n_des_;
    /* number of LMIdx (l, m) */
    int n_lm_all_;
    int n_lm_half_;

    // Kokkos
    dview_1d reg_coeffs_kk_ = dview_1d("reg_coeffs_kk_", 0);

    /* flatten_lm_array_: IrrepsTermIdx -> LMIdx */
    StaticCrsGraph d_lm_array_;
    /* lm_coeffs_kk_: IrrepsTermIdx -> double */
    dview_1d lm_coeffs_kk_ = dview_1d("lm_coeffs_kk_", 0);
    /* d_irreps_type_combs_(itcidx: IrrepsTypeCombIdx) is list of TypeCombs */
    StaticCrsGraph d_irreps_type_combs_;
    /*
    irreps_type_intersection(itcidx: IrrepsTypeCombIdx, type: ElementType) is true
    iff all `type_combs` contain `type`.
    */
    dview_2b irreps_type_intersection_ = dview_2b("irreps_type_intersection_", 0, 0);
    /* irreps_type_mapping_: IrrepsTypeCombIdx -> IrrepsIdx */
    dview_1i irreps_type_mapping_ = dview_1i("irreps_type_mapping_", 0);
    /* irreps_first_term_: IrrepsIdx -> IrrepsTermIdx */
    dview_1i irreps_first_term_ = dview_1i("irreps_first_term_", 0);
    /* irrreps_num_terms: IrrepsIdx -> int */
    dview_1i irreps_num_terms_ = dview_1i("irreps_num_terms_", 0);
    StaticCrsGraph d_polynomial_index_;

    /*
    other_type_[TypeCombIdx][type1] returns type2 if exists.
    If not exist, return -1
    */
    dview_2i other_type_kk_ = dview_2i("other_type_kk_", 0, 0);
    /* type_pairs_: (type1, type2) -> TypeCombIdx */
    dview_2i type_pairs_kk_ = dview_2i("type_pairs_kk_", 0, 0);

    dview_2i lm_info_kk_ = dview_2i("lm_info_kk_", 0, 0);
    /* lm2l_: LMIdx for (l, m) -> l */
    dview_1i lm2l_ = dview_1i("lm2l_", 0);
    /* lm2m_: LMIdx for (l, m) -> m */
    dview_1i lm2m_ = dview_1i("lm2m_", 0);
    /* precomputed coefficients for spherical harmonics, a_{l}^{m}*/
    dview_1d sph_coeffs_A_ = dview_1d("sph_coeffs_A_", 0);
    /* precomputed coefficients for spherical harmonics, b_{l}^{m}*/
    dview_1d sph_coeffs_B_ = dview_1d("sph_coeffs_B_", 0);

    /* ---------------------------------------------------------------------- */
    // temporal
    /* ---------------------------------------------------------------------- */
    /* number of atoms */
    int inum_;
    /* number of atom pairs in half neighbor list */
    int n_pairs_;
    /* neighbor_pair_index_[npidx: NeighborPairIdx] = (i, j) */
    dview_1p neighbor_pair_index_kk_ = dview_1p("neighbor_pair_index_kk_", 0);
    /* neighbor_pair_displacements_[npidx: NeighborPairIdx][3] */
    dview_2d neighbor_pair_displacements_kk_ = dview_2d("neighbor_pair_displacements_kk_", 0, 0);
    /* neighbor_pair_typecomb_kk_: NeighborPairIdx -> TypeCombIdx */
    dview_1i neighbor_pair_typecomb_kk_ = dview_1i("neighbor_pair_typecomb_kk_", 0);
    /* types_kk_: NeighborPairIdx -> ElementType */
    dview_1i types_kk_ = dview_1i("types_kk_", 0);

    view_1d d_distance_= view_1d("d_distance_", 0);
    view_2d d_fn_= view_2d("d_fn_", 0, 0);
    view_2d d_fn_der_= view_2d("d_fn_der_", 0, 0);
    /* normalized associated Legendre polynomial (ALP) for neighbors */
    view_2d d_alp_ = view_2d("d_alp_", 0, 0);
    /* normalized associated Legendre polynomial (ALP) devided by sin theta for neighbors */
    view_2d d_alp_sintheta_ = view_2d("d_alp_sintheta_", 0, 0);
    view_2dc d_ylm_= view_2dc("ylm_kk_", 0, 0);
    view_2dc d_ylm_dx_ = view_2dc("d_ylm_dx_", 0, 0);
    view_2dc d_ylm_dy_ = view_2dc("d_ylm_dy_", 0, 0);
    view_2dc d_ylm_dz_ = view_2dc("d_ylm_dz_", 0, 0);

    view_4d d_anlm_r_ = view_4d("d_anlm_r_", 0, 0, 0, 0);
    view_4d d_anlm_i_ = view_4d("d_anlm_i_", 0, 0, 0, 0);
    /* order parameters, (inum_, n_types, n_fn_, n_lm_all) */
    view_4dc d_anlm_= view_4dc("d_anlm_", 0, 0, 0, 0);
    /* O(3)-invariant features, (inum_, n_des_) */
    dview_2d structural_features_kk_ = dview_2d("structural_features_kk_", 0, 0);
    /* polynomial_adjoints_[i][fidx: FeatureIdx] adjoint of structural feature `fidx` of atom i */
    view_2d d_polynomial_adjoints_ = view_2d("d_polynomial_adjoints_", 0, 0);
    /* basis_function_adjoints_[i][TypeCombIdx][n][LMIdx] */
    view_4dc d_basis_function_adjoints_ = view_4dc("d_basis_function_adjoints_", 0, 0, 0, 0);

    double energy_;
    /* atom-wise energy, (inum_, ) */
    dview_1d site_energy_kk_ = dview_1d("site_energy_kk_", 0);
    /* forces, (inum_, 3) */
    dview_2d forces_kk_ = dview_2d("forces_kk_", 0, 0);
    /* stress tensor in voigt order, (6, ) */
    dview_1d stress_kk_ = dview_1d("stress_kk_", 6);

public:
    MLIPModel() = default;
    ~MLIPModel() = default;

    void initialize(const MLIPInput& input, const vector1d& reg_coeffs, const Readgtinv& gtinvdata);

    // getters
    vector2d get_structural_features();
    vector2d get_polynomial_features();
    double get_energy() const { return energy_; };
    vector1d get_site_energies();
    vector2d get_forces();
    vector1d get_stress();

    // setter for structure
    void set_structure(const std::vector<ElementType>& types,
                       const vector3d& displacements, const std::vector<std::vector<SiteIdx>>& neighbors);
    void compute();
    void prepare_features();

    // Do not call these functions outside!
    void compute_basis_functions();
    void compute_order_parameters();
    void compute_structural_features();
    void compute_basis_function_adjoints();
    void compute_basis_functions_der();
    void compute_energy();
    void compute_forces_and_stress();
    void compute_polynomial_adjoints();

    KOKKOS_INLINE_FUNCTION
    double product_real_part(const Kokkos::complex<double>& lhs, const Kokkos::complex<double>& rhs) const;

    void dump(std::ostream& os) const;

private:
    void initialize_radial_basis(const char* pair_type, const vector2d& params);
    void initialize_gtinv(const int n_types, const int n_fn, const int model_type, const int maxp,
                          const Readgtinv& gtinvdata);
    void initialize_typecomb(const int n_types, const int n_typecomb);
    void initialize_sph(const int maxl);
    void initialize_sph_AB(const int maxl);

};

} // namespace MLIP

#endif
