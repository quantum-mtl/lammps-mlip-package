#ifndef MLIPKK_IRREPS_TYPE_H_
#define MLIPKK_IRREPS_TYPE_H_

#include <iostream>
#include <vector>
#include <utility>

#include "mlipkk_types.h"

namespace MLIP_NS {

/// @brief equivalent to `LinearTermGtinv`
struct IrrepsTypePair {
    IrrepsIdx irreps_idx;
    std::vector<TypeCombIdx> type_combs;
    /* type_intersection[type] is true iff all `type_combs` contain `type` */
    std::vector<bool> type_intersection;

    void dump(std::ostream& os) const;
};

std::vector<bool> get_type_intersection(const int n_types,
                                        const std::vector<std::pair<ElementType, ElementType>>& type_pairs_mapping,
                                        const std::vector<TypeCombIdx>& type_comb);

std::vector<std::pair<ElementType, ElementType>> get_type_pairs_mapping(const int n_types);

std::vector<IrrepsTypePair> get_unique_irreps_type_pairs(int n_types,
                                                         const vector2i& l_array);

} // namespace MLIP_NS

#endif // MLIPKK_IRREPS_TYPE_H_
