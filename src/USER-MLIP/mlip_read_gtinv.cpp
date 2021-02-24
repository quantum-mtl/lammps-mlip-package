/****************************************************************************

        Copyright (C) 2020 Atsuto Seko
                seko@cms.mtl.kyoto-u.ac.jp

        This program is free software; you can redistribute it and/or
        modify it under the terms of the GNU General Public License
        as published by the Free Software Foundation; either version 2
        of the License, or (at your option) any later version.

        This program is distributed in the hope that it will be useful,
        but WITHOUT ANY WARRANTY; without even the implied warranty of
        MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
        GNU General Public License for more details.

        You should have received a copy of the GNU General Public License
        along with this program; if not, write to
        the Free Software Foundation, Inc., 51 Franklin Street,
        Fifth Floor, Boston, MA 02110-1301, USA, or see
        http://www.gnu.org/copyleft/gpl.txt

*****************************************************************************/

#include "mlip_read_gtinv.h"

Readgtinv::Readgtinv(const int& gtinv_order, const vector1i& gtinv_maxl,
                     const std::vector<bool>& gtinv_sym, const int& n_type){
    screening(gtinv_order, gtinv_maxl, gtinv_sym, n_type);
}

/// @brief collect required info from GtinvData
void Readgtinv::screening(const int& gtinv_order, const vector1i& gtinv_maxl,
                          const std::vector<bool>& gtinv_sym, const int& n_type){

    MLIP_NS::GtinvData data;
    const vector2i l_array_all = data.get_l_array();
    const vector3i m_array_all = data.get_m_array();
    const vector2d coeffs_all = data.get_coeffs();

    for (int i = 0; i < l_array_all.size(); ++i){
        const vector1i &lcomb = l_array_all[i];
        bool tag(true);
        const int order = lcomb.size();
        const int maxl = *(lcomb.end()-1);
        if (order > 1){
            if (order > gtinv_order or maxl > gtinv_maxl[order-2]) tag = false;
            if (gtinv_sym[order-2] == true){
                int n_ele = std::count(lcomb.begin(), lcomb.end(), lcomb[0]);
                if (n_ele != order) tag = false;
            }
        }

        if (tag == true){
            int l, m;
            vector2i vec1(m_array_all[i].size(), vector1i(order));
            for (int j = 0; j < m_array_all[i].size(); ++j){
                const auto &mcomb = m_array_all[i][j];
                for (int k = 0; k < order; ++k){
                    l = lcomb[k], m = mcomb[k];
                    vec1[j][k] = l*l+l+m;
                }
            }
            l_array.emplace_back(lcomb);
            lm_array.emplace_back(vec1);
            coeffs.emplace_back(coeffs_all[i]);
        }
    }
}

const vector3i& Readgtinv::get_lm_seq() const{ return lm_array; }
const vector2i& Readgtinv::get_l_comb() const{ return l_array; }
const vector2d& Readgtinv::get_lm_coeffs() const{ return coeffs; }
