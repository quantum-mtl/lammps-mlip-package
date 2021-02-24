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

	    Header file for Readgtinv.cpp

****************************************************************************/

#ifndef __MLIP_READ_GTINV
#define __MLIP_READ_GTINV

#include "mlip_gtinv_data.h"
#include "mlip_pymlcpp.h"

class Readgtinv{
    vector2i l_array;
    vector3i lm_array;
    vector2d coeffs;

public:
    Readgtinv() = default;
    Readgtinv(const int& gtinv_order, const vector1i& gtinv_maxl,
              const std::vector<bool>& gtinv_sym, const int& n_type);
   ~Readgtinv() = default;

    const vector3i& get_lm_seq() const;
    const vector2i& get_l_comb() const;
    const vector2d& get_lm_coeffs() const;

private:
    void screening(const int& gtinv_order, const vector1i& gtinv_maxl,
                   const std::vector<bool>& gtinv_sym, const int& n_type);
};

#endif
