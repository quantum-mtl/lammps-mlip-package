/****************************************************************************

        Copyright (C) 2018 Atsuto Seko
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

#ifndef __PYMLCPP_MODEL
#define __PYMLCPP_MODEL

#include <vector>
#include <complex>

namespace MLIP_NS {

using SiteIdx = int;
// index for (i, j) neighbor pair
using NeighborPairIdx = int;

/* index for type of elements */
using ElementType = int;
/* index for unordered pair of ElementType, (ElementType, ElementType) */
using TypeCombIdx = int;

/* index for m <= 0, (l, m) = (0, 0), (1, -1), (1, 0), (2, -2), ... */
using LMInfoIdx = int;
/* index for all (l, m), (l, m) = (0, 0), (1, -1), (1, 0), (1, 1), ... */
using LMIdx = int;

/* index for Irreps (list of l, sigma) */
using IrrepsIdx = int;
/* index for each term in Irreps */
using IrrepsTermIdx = int;

/* index for tuple of (IrrepsIdx, TypeCombIdx) */
using IrrepsTypeCombIdx = int;
/* index for structural features, (IrrepsTypeCombIdx, index of radial basis function)*/
using FeatureIdx = int;
/* index for polynomial features */
using PolynomialIdx = int;

using vector1i = std::vector<int>;
using vector2i = std::vector<vector1i>;
using vector3i = std::vector<vector2i>;
using vector4i = std::vector<vector3i>;

using vector1d = std::vector<double>;
using vector2d = std::vector<vector1d>;
using vector3d = std::vector<vector2d>;
using vector4d = std::vector<vector3d>;
using vector5d = std::vector<vector4d>;

using dc = std::complex<double>;
using vector1dc = std::vector<dc>;
using vector2dc = std::vector<vector1dc>;
using vector3dc = std::vector<vector2dc>;
using vector4dc = std::vector<vector3dc>;
using vector5dc = std::vector<vector4dc>;

} // namespace MLIP_NS
#endif
