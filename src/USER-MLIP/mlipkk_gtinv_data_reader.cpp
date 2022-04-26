#include "mlipkk_gtinv_data_reader.h"

#include <algorithm>

#include "mlipkk_gtinv_data.h"

namespace MLIP_NS {

Readgtinv::Readgtinv(const int gtinv_order, const vector1i& gtinv_maxl,
                     const std::vector<bool>& gtinv_sym) {
    screening(gtinv_order, gtinv_maxl, gtinv_sym);
}

/// @brief collect required info from GtinvDataKK
/// @param[in] gtinv_order maximum angular order to be used
/// @param[in] gtinv_maxl vector of l_max for order = 2, 3, ...
/// @param[in] gtinv_sym
void Readgtinv::screening(const int& gtinv_order, const vector1i& gtinv_maxl,
                          const std::vector<bool>& gtinv_sym) {
    MLIP_NS::GtinvDataKK data;
    const vector2i l_array_all = data.get_l_array();
    const vector3i m_array_all = data.get_m_array();
    const vector2d coeffs_all = data.get_coeffs();

    IrrepsIdx count = 0;
    IrrepsTermIdx iterm_count = 0;
    for (int i = 0; i < static_cast<int>(l_array_all.size()); ++i) {
        const vector1i& lcomb = l_array_all[i];
        bool is_required(true);  // flag whether to fetch from GtinvDataKK
        const int order =
            static_cast<int>(lcomb.size());  // number of angular momemtums
        const int maxl = *(lcomb.end() - 1);
        if (order > 1) {
            if ((order > gtinv_order) or (maxl > gtinv_maxl[order - 2])) {
                is_required = false;
            }
            if (gtinv_sym[order - 2] == true) {
                int n_ele = static_cast<int>(
                    std::count(lcomb.begin(), lcomb.end(), lcomb[0]));
                if (n_ele != order) {
                    is_required = false;
                }
            }
        }

        if (is_required == true) {
            const int num_terms = static_cast<int>(m_array_all[i].size());
            std::vector<std::vector<LMIdx>> vec1(num_terms,
                                                 std::vector<LMIdx>(order));
            for (int term = 0; term < num_terms; ++term) {
                const auto& mcomb = m_array_all[i][term];
                for (int k = 0; k < order; ++k) {
                    const int l = lcomb[k];
                    const int m = mcomb[k];
                    const LMIdx lm = l * l + l + m;
                    vec1[term][k] = lm;
                }
                flatten_lm_array_.emplace_back(vec1[term]);
                flatten_coeffs_.emplace_back(coeffs_all[i][term]);
                irreps_term_mapping_.emplace_back(count);
                if (term == 0) {
                    // first term
                    irreps_first_term_.emplace_back(iterm_count);
                }
                ++iterm_count;
            }

            l_array.emplace_back(lcomb);
            lm_array.emplace_back(vec1);
            coeffs.emplace_back(coeffs_all[i]);
            irreps_num_terms_.emplace_back(num_terms);

            ++count;
        }
    }
}

}  // namespace MLIP_NS
