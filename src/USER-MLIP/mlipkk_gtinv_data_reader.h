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

	    Header file for GtinvData.cpp

****************************************************************************/

#ifndef MLIPKK_GTINV_DATA_READER_H_
#define MLIPKK_GTINV_DATA_READER_H_

#include <vector>

#include "mlipkk_types.h"

namespace MLIP_NS {

class Readgtinv{
    vector2i l_array;
    vector3i lm_array;  // TODO
    vector2d coeffs;

    /* flatten_lm_array_: IrrepsTermIdx -> [LMIdx] */
    vector2i flatten_lm_array_;
    /* flatten_coeffs_: IrrepsTermIdx -> double */
    vector1d flatten_coeffs_;
    /* irreps_term_mapping_: IrrepsTermIdx -> IrrepsIdx */
    std::vector<IrrepsIdx> irreps_term_mapping_;
    /*
    return the first IrrepsIdx within terms,
    irreps_first_term_: IrrepsIdx -> IrrepsTermIdx
    */
    std::vector<IrrepsTermIdx> irreps_first_term_;
    vector1i irreps_num_terms_;

public:
    Readgtinv() = default;
    Readgtinv(const int gtinv_order, const vector1i& gtinv_maxl,
              const std::vector<bool>& gtinv_sym);
   ~Readgtinv() = default;

    /*
    Table of sets of angular numbers corresponding to all symmetry Irreps to be used.
    Note that some angular sets (e.g. {1, 1, 1, 1}) are duplicated because
    there may be several all symmetry Irreps with the same angular sets!
    l_array: IrrepsIdx(int) -> {l1, l2, ...}
    */
    const vector2i& get_l_comb() const { return l_array; };
    /*
    lm_array: IrrepsIdx(int) -> [# of terms] -> {LMIdx for m1, LMIdx for m2, ...}
    where LMIdx is index for (l, m) = (0, 0), (1, -1), (1, 0), (1, 1), ...
    */
    const vector3i& get_lm_seq() const { return lm_array; };
    const vector2i& get_flatten_lm_seq() const { return flatten_lm_array_; };
    /*
    coeffs_all[i] is table of generalized Clebsch-Gordon coefficients for l_array_all[i]
    coeffs_all: IrrepsIdx(int) -> vector<double>
    */
    const vector2d& get_lm_coeffs() const { return coeffs; };
    /* flatten_coeffs_: IrrepsTermIdx -> double */
    const vector1d& get_flatten_lm_coeffs() const { return flatten_coeffs_; };

    /* irreps_term_mapping_: IrrepsTermIdx -> IrrepsIdx */
    const std::vector<IrrepsIdx> get_irreps_term_mapping() const { return irreps_term_mapping_; };
    const vector1i& get_irreps_first_term() const { return irreps_first_term_; };
    const vector1i& get_irreps_num_terms() const { return irreps_num_terms_; };

private:
    void screening(const int& gtinv_order, const vector1i& gtinv_maxl,
                   const std::vector<bool>& gtinv_sym);
};

} // namespace MLIP_NS
#endif // MLIPKK_GTINV_DATA_READER_H_
