#ifndef MLIPKK_POLYNOMIAL_H_
#define MLIPKK_POLYNOMIAL_H_

#include <vector>
#include <map>

#include "mlipkk_irreps_type.h"

namespace MLIP_NS {

class MLIPPolynomial {
    int maxp_;
    int n_fn_;
    int n_types_;
    /* number of pair of Irreps and TypeComb */
    int n_itc_;
    /* sorted as (IrrepsTypeCombs for n = 0), (IrrepsTypeCombs for n = 1), ... */
    std::vector<std::vector<FeatureIdx>> polynomial_index_;
    /*
    polynomial_typecomb_intersection_[type: ElementType][pidx: PolynomialIdx] is true
    iff all structural features in `pidx` have `type`.
    */
    std::vector<std::vector<bool>> polynomial_typecomb_intersection_;
    /* map list of FeatureIdx to PolynomialIdx */
    std::map<std::vector<FeatureIdx>, PolynomialIdx> polynomial_index_mapping_;
    /*
    structural_features_prod_index_[fidx: FeatureIdx] is list of PolynomialIdx with `poly_order` length
    used for computing derivative of polynomial features.
    if a polynomial feature consists from one structural feature, return -1
    */
    std::vector<std::vector<FeatureIdx>> structural_features_prod_index_;

public:
    MLIPPolynomial() = default;
    ~MLIPPolynomial() = default;
    MLIPPolynomial(const int model_type, const int maxp, const int n_fn, const int n_types,
                   const std::vector<IrrepsTypePair>& irreps_type_pairs);

    const std::vector<std::vector<FeatureIdx>>& get_polynomial_index() const { return polynomial_index_; };
    const std::vector<std::vector<bool>>& get_polynomial_typecomb_intersection() const { return polynomial_typecomb_intersection_; };
    /*
    structural_features_prod_index_[fidx: FeatureIdx] is list of PolynomialIdx with `poly_order` length
    used for computing derivative of polynomial features.
    if a polynomial feature consists from one structural feature, return -1
    */
    const std::vector<std::vector<FeatureIdx>>& get_structural_features_prod_index() const { return structural_features_prod_index_; };

    IrrepsTypeCombIdx get_irreps_type_idx(const FeatureIdx& fidx) const;
    int get_fn_idx(const FeatureIdx& fidx) const;
    FeatureIdx get_feature_idx(const IrrepsTypeCombIdx& itcidx, const int n) const;
    int get_n_des() const { return n_fn_ * n_itc_; };
    int get_n_itc() const { return n_itc_; };

};



} // namespace MLIP_NS

#endif //MLIPKK_POLYNOMIAL_H_
