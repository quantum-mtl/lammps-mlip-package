#include "mlipkk_irreps_type.h"

#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <utility>

#include "mlipkk_utils.h"

namespace MLIP_NS {

void IrrepsTypePair::dump(std::ostream& os) const {
    os << "IrrepsIdx: " << irreps_idx << ", type_combs: (";
    for (const auto& tc: type_combs) {
        os << tc << ", ";
    }
    os << ")" << std::endl;
}


std::vector<bool> get_type_intersection(const int n_types,
                                        const std::vector<std::pair<ElementType, ElementType>>& type_pairs_mapping,
                                        const std::vector<TypeCombIdx>& type_comb)
{
    std::vector<bool> intersection(n_types, true);
    for (TypeCombIdx idx : type_comb) {
        auto p = type_pairs_mapping[idx];
        for (ElementType type = 0; type < n_types; ++type) {
            if ((type != p.first) && (type != p.second)) {
                intersection[type] = false;
            }
        }
    }
    return intersection;
}

std::vector<std::pair<ElementType, ElementType>> get_type_pairs_mapping(const int n_types) {
    std::vector<std::pair<ElementType, ElementType>> type_pairs_mapping;
    for (ElementType type1 = 0; type1 < n_types; ++type1) {
        for (ElementType type2 = type1; type2 < n_types; ++type2) {
            type_pairs_mapping.emplace_back(std::make_pair(type1, type2));
        }
    }
    return type_pairs_mapping;
}

std::vector<IrrepsTypePair> get_unique_irreps_type_pairs(int n_types,
                                                         const vector2i& l_array)
{
    int num_type_combs = n_types * (n_types + 1) / 2;
    const auto type_pairs_mapping = get_type_pairs_mapping(n_types);

    int max_order = l_array[static_cast<int>(l_array.size()) - 1].size();
    std::vector<std::vector<std::vector<TypeCombIdx>>> uniq_type_combs(max_order);
    for (int order = 1; order <= max_order; ++order) {
        std::vector<std::vector<TypeCombIdx>> all_combs = get_combinations_with_repetition(num_type_combs, order);
        for (auto comb: all_combs) {
            // check if intersection of types is not empty
            auto intersection = get_type_intersection(n_types, type_pairs_mapping, comb);
            if (std::any_of(intersection.begin(), intersection.end(), [](bool e){ return e; })) {
                uniq_type_combs[order - 1].emplace_back(comb);
            }
        }
    }

    std::vector<IrrepsTypePair> irreps_types_pairs;
    for (IrrepsIdx i = 0; i < static_cast<int>(l_array.size()); ++i) {
        const auto& lcomb = l_array[i];  // already sorted, e.g. {1, 1, 2}
        std::set<std::multiset<std::pair<int, TypeCombIdx>>> set_irreps_types;

        int order = static_cast<int>(lcomb.size());
        for (const auto& sorted_type_comb: uniq_type_combs[order - 1]) {
            std::vector<TypeCombIdx> type_comb(sorted_type_comb);
            do {
                std::multiset<std::pair<int, TypeCombIdx>> irreps_types;
                for (int j = 0; j < order; ++j) {
                    irreps_types.insert(std::make_pair(lcomb[j], type_comb[j]));
                }
                set_irreps_types.insert(irreps_types);
            } while(std::next_permutation(type_comb.begin(), type_comb.end()));
        }

        for (const auto& irreps_types: set_irreps_types) {
            std::vector<TypeCombIdx> type_comb;
            for (const auto& lt: irreps_types) {
                type_comb.emplace_back(lt.second);
            }
            auto intersection = get_type_intersection(n_types, type_pairs_mapping, type_comb);
            irreps_types_pairs.emplace_back(IrrepsTypePair{i, type_comb, intersection});
        }

    }
    return irreps_types_pairs;
}

} // namespace MLIP_NS
