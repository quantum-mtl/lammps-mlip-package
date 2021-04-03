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

 ******************************************************************************/

#ifndef __MLIP_BASIS_FUNCTION_HPP
#define __MLIP_BASIS_FUNCTION_HPP

namespace MLIP_NS {

double cosine_cutoff_function(const double& dis, const double& cutoff);
double cosine_cutoff_function_d(const double& dis, const double& cutoff);

double gauss(const double& dis, const double& beta, const double& mu);
void gauss_d(const double& dis, const double& beta, const double& mu,
             double& bf, double& bf_d);

} // namespace MLIP_NS
#endif
