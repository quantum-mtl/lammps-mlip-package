#include "mlipkk_polynomial.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "mlipkk_irreps_type.h"
#include "mlipkk_utils.h"

namespace MLIP_NS {

MLIPPolynomial::MLIPPolynomial(
    const int model_type, const int maxp, const int n_fn, const int n_types,
    const std::vector<IrrepsTypePair>& irreps_type_pairs)
    : maxp_(maxp),
      n_fn_(n_fn),
      n_types_(n_types),
      n_itc_(static_cast<int>(irreps_type_pairs.size())) {
    polynomial_index_.clear();
    polynomial_index_mapping_.clear();
    polynomial_typecomb_intersection_.resize(n_types);

    // set polynomial_index_, polynomial_index_mapping,
    // polynomial_typecomb_intersection_, and n_coeffs_
    if (model_type == 1) {
        enumerate_polynomials_model_type_1(irreps_type_pairs);
    } else if (model_type == 2) {
        enumerate_polynomials_model_type_2(irreps_type_pairs);
    } else if (model_type == 3) {
        enumerate_polynomials_model_type_3(irreps_type_pairs);
    } else if (model_type == 4) {
        enumerate_polynomials_model_type_4(irreps_type_pairs);
    } else {
        std::cerr << "Unknown model_type: " << model_type << std::endl;
        exit(1);
    }
}

IrrepsTypeCombIdx MLIPPolynomial::get_irreps_type_idx(
    const FeatureIdx& fidx) const {
    return fidx % n_itc_;
}

int MLIPPolynomial::get_fn_idx(const FeatureIdx& fidx) const {
    return fidx / n_itc_;
}

FeatureIdx MLIPPolynomial::get_feature_idx(const IrrepsTypeCombIdx& itcidx,
                                           const int n) const {
    return n * n_itc_ + itcidx;
}

void MLIPPolynomial::dump(std::ostream& os, const FeatureIdx fidx) const {
    const IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
    const int n = get_fn_idx(fidx);
    os << "fidx = " << fidx << ": itcidx = " << itcidx << ", n = " << n
       << std::endl;
}

/// only consider repeated FeatureIdx
void MLIPPolynomial::enumerate_polynomials_model_type_1(
    const std::vector<IrrepsTypePair>& irreps_type_pairs) {
    std::cerr << "Warning: model_type=1 is not tested!" << std::endl;
    int n_des = n_itc_ * n_fn_;

    int n_poly = 0;
    for (int poly_order = 1; poly_order <= maxp_; ++poly_order) {
        for (FeatureIdx fidx = 0; fidx < n_des; ++fidx) {
            std::vector<FeatureIdx> fcomb;
            for (int i = 0; i < poly_order; ++i) {
                fcomb.emplace_back(fidx);
            }

            // check if intersection of type combs is not empty
            IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
            const auto& intersection =
                irreps_type_pairs[itcidx].type_intersection;
            if (std::any_of(intersection.begin(), intersection.end(),
                            [](const bool e) { return e; })) {
                polynomial_index_.emplace_back(fcomb);
                for (ElementType type = 0; type < n_types_; ++type) {
                    polynomial_typecomb_intersection_[type].emplace_back(
                        intersection[type]);
                }
                polynomial_index_mapping_[fcomb] = n_poly++;
            }
        }
    }
    n_coeffs_ = n_poly;
}

void MLIPPolynomial::enumerate_polynomials_model_type_2(
    const std::vector<IrrepsTypePair>& irreps_type_pairs) {
    int n_des = n_itc_ * n_fn_;

    int n_poly = 0;
    for (int poly_order = 1; poly_order <= maxp_; ++poly_order) {
        // do not use get_combinations_with_repetition for backward
        // compatibility!
        auto all_combs =
            get_combinations_with_repetition_gtinv(n_des, poly_order);
        for (std::vector<FeatureIdx> fcomb : all_combs) {
            // check if intersection of type combs is not empty
            const auto intersection =
                get_typecombs_intersection(fcomb, irreps_type_pairs);
            if (std::any_of(intersection.begin(), intersection.end(),
                            [](const bool e) { return e; })) {
                polynomial_index_.emplace_back(fcomb);
                for (ElementType type = 0; type < n_types_; ++type) {
                    polynomial_typecomb_intersection_[type].emplace_back(
                        intersection[type]);
                }
                polynomial_index_mapping_[fcomb] = n_poly++;
            }
        }
    }
    n_coeffs_ = n_poly;
}

void MLIPPolynomial::enumerate_polynomials_model_type_3(
    const std::vector<IrrepsTypePair>& irreps_type_pairs) {
    int n_des = n_itc_ * n_fn_;

    // gather FeatureIdx with gtinv_order == 1
    std::vector<FeatureIdx> pair_feature_indices;
    for (FeatureIdx fidx = 0; fidx < n_des; ++fidx) {
        const IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
        const int gtinv_order =
            static_cast<int>(irreps_type_pairs[itcidx].type_combs.size());
        if (gtinv_order == 1) {
            pair_feature_indices.emplace_back(fidx);
        }
    }

    enumerate_polynomials_model_type_3_or_4(irreps_type_pairs,
                                            pair_feature_indices);
}

void MLIPPolynomial::enumerate_polynomials_model_type_4(
    const std::vector<IrrepsTypePair>& irreps_type_pairs) {
    int n_des = n_itc_ * n_fn_;

    // gather FeatureIdx with gtinv_order == 1 or 2
    std::vector<FeatureIdx> lower_feature_indices;
    for (FeatureIdx fidx = 0; fidx < n_des; ++fidx) {
        const IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
        const int gtinv_order =
            static_cast<int>(irreps_type_pairs[itcidx].type_combs.size());
        if (gtinv_order <= 2) {
            lower_feature_indices.emplace_back(fidx);
        }
    }

    enumerate_polynomials_model_type_3_or_4(irreps_type_pairs,
                                            lower_feature_indices);
}

void MLIPPolynomial::enumerate_polynomials_model_type_3_or_4(
    const std::vector<IrrepsTypePair>& irreps_type_pairs,
    const std::vector<FeatureIdx>& lower_feature_indices) {
    int n_des = n_itc_ * n_fn_;

    int n_poly = 0;

    // order-1 terms
    for (FeatureIdx fidx = 0; fidx < n_des; ++fidx) {
        std::vector<FeatureIdx> fcomb = {fidx};

        IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
        const auto& intersection = irreps_type_pairs[itcidx].type_intersection;

        polynomial_index_.emplace_back(fcomb);
        for (ElementType type = 0; type < n_types_; ++type) {
            polynomial_typecomb_intersection_[type].emplace_back(
                intersection[type]);
        }
        polynomial_index_mapping_[fcomb] = n_poly++;
    }

    // for poly_order>=2, only consider lower_feature_indices
    for (int poly_order = 2; poly_order <= maxp_; ++poly_order) {
        // do not use get_combinations_with_repetition for backward
        // compatibility!
        auto all_mapping_combs = get_combinations_with_repetition_gtinv(
            static_cast<int>(lower_feature_indices.size()), poly_order);
        for (std::vector<int> choices : all_mapping_combs) {
            std::vector<FeatureIdx> fcomb;
            for (int idx : choices) {
                fcomb.emplace_back(lower_feature_indices[idx]);
            }

            // check if intersection of type combs is not empty
            const auto intersection =
                get_typecombs_intersection(fcomb, irreps_type_pairs);
            if (std::any_of(intersection.begin(), intersection.end(),
                            [](const bool e) { return e; })) {
                polynomial_index_.emplace_back(fcomb);
                for (ElementType type = 0; type < n_types_; ++type) {
                    polynomial_typecomb_intersection_[type].emplace_back(
                        intersection[type]);
                }
                polynomial_index_mapping_[fcomb] = n_poly++;
            }
        }
    }

    n_coeffs_ = n_poly;
}

std::vector<bool> MLIPPolynomial::get_typecombs_intersection(
    const std::vector<FeatureIdx>& fcomb,
    const std::vector<IrrepsTypePair>& irreps_type_pairs) {
    std::vector<bool> intersection(n_types_, true);
    for (FeatureIdx fidx : fcomb) {
        IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
        const auto& intersection_fidx =
            irreps_type_pairs[itcidx].type_intersection;
        for (ElementType type = 0; type < n_types_; ++type) {
            intersection[type] =
                (intersection[type] && intersection_fidx[type]);
        }
    }
    return intersection;
}

}  // namespace MLIP_NS
