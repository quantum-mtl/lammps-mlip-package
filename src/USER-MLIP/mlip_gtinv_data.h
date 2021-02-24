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

#ifndef __MLIP_GTINV_DATA
#define __MLIP_GTINV_DATA

#include "mlip_pymlcpp.h"

namespace MLIP_NS {

class GtinvData{
    /*
    Table of sets of angular numbers corresponding to all symmetry Irreps.
    Note that some angular sets (e.g. {1, 1, 1, 1}) are duplicated because
    there may be several all symmetry Irreps with the same angular sets!
    l_array_all: IrrepsIdx(int) -> {l1, l2, ...}
    */
    vector2i l_array_all;
    /* m_array_all[i] is table of m-terms for l_array_all[i] */
    vector3i m_array_all;
    /*
    coeffs_all[i] is table of generalized Clebsch-Gordon coefficients for l_array_all[i]
    */
    vector2d coeffs_all;

public:
    GtinvData();
   ~GtinvData() = default;

    const vector2i& get_l_array() const;
    const vector3i& get_m_array() const;
    const vector2d& get_coeffs() const;

private:
    void set_gtinv_info();
};

} // namespace MLIP_NS
#endif
