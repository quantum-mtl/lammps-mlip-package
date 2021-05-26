#include "mlipkk_polynomial.h"

#include <vector>
#include <algorithm>
#include <cassert>

#include "mlipkk_irreps_type.h"
#include "mlipkk_utils.h"

namespace MLIP_NS {

MLIPPolynomial::MLIPPolynomial(const int model_type, const int maxp, const int n_fn,
                               const int n_types,
                               const std::vector<IrrepsTypePair>& irreps_type_pairs)
                               : maxp_(maxp), n_fn_(n_fn), n_types_(n_types),
                                 n_itc_(static_cast<int>(irreps_type_pairs.size()))
{
    // TODO: model_type other than 2
    assert(model_type == 2);

    polynomial_index_.clear();
    polynomial_index_mapping_.clear();
    // polynomial_typecomb_intersection_ = std::vector<std::vector<bool>>(n_types_);
    polynomial_typecomb_intersection_.resize(n_types);

    int n_des = n_itc_ * n_fn_;

    int n_poly = 0;
    for (int poly_order = 1; poly_order <= maxp; ++poly_order) {
        // do not use get_combinations_with_repetition for backward compatibility!
        auto all_combs = get_combinations_with_repetition_gtinv(n_des, poly_order);
        for (std::vector<FeatureIdx> fcomb: all_combs) {
            // check if intersection of type combs is not empty
            std::vector<bool> intersection(n_types, true);
            for (FeatureIdx fidx: fcomb) {
                IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
                const auto& intersection_fidx = irreps_type_pairs[itcidx].type_intersection;
                for (ElementType type = 0; type < n_types_; ++type) {
                    intersection[type] = (intersection[type] && intersection_fidx[type]);
                }
            }
            if (std::any_of(intersection.begin(), intersection.end(), [](const bool e){ return e; })) {
                polynomial_index_.emplace_back(fcomb);
                for (ElementType type = 0; type < n_types_; ++type) {
                    polynomial_typecomb_intersection_[type].emplace_back(intersection[type]);
                }
                polynomial_index_mapping_[fcomb] = n_poly++;
            }
        }
    }

    structural_features_prod_index_.resize(n_poly);
    for (PolynomialIdx pidx = 0; pidx < n_poly; ++pidx) {
        const int poly_order = static_cast<int>(polynomial_index_[pidx].size());
        if (poly_order == 1) {
            structural_features_prod_index_[pidx].emplace_back(-1);
        } else {
            for (int p1 = 0; p1 < poly_order; ++p1) {
                std::vector<FeatureIdx> fcomb;
                for (int p2 = 0; p2 < poly_order; ++p2) {
                    if (p2 == p1) {
                        continue;
                    }
                    fcomb.emplace_back(polynomial_index_[pidx][p2]);
                }
                // FeaturesIdx are sorted in descending order in polynomial feature
                std::sort(fcomb.rbegin(), fcomb.rend());

                PolynomialIdx pidx_dev = polynomial_index_mapping_[fcomb];
                structural_features_prod_index_[pidx].emplace_back(pidx_dev);
            }
        }
    }
}

IrrepsTypeCombIdx MLIPPolynomial::get_irreps_type_idx(const FeatureIdx& fidx) const {
    return fidx % n_itc_;
}

int MLIPPolynomial::get_fn_idx(const FeatureIdx& fidx) const {
    return fidx / n_itc_;
}

FeatureIdx MLIPPolynomial::get_feature_idx(const IrrepsTypeCombIdx& itcidx, const int n) const {
    return n * n_itc_ + itcidx;
}

void MLIPPolynomial::dump(std::ostream& os, const FeatureIdx fidx) const {
    const IrrepsTypeCombIdx itcidx = get_irreps_type_idx(fidx);
    const int n = get_fn_idx(fidx);
    os << "fidx = " << fidx << ": itcidx = " << itcidx << ", n = " << n << std::endl;
}

} // namespace MLIP_NS
