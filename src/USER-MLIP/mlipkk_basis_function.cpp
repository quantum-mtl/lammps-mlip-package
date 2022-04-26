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

#include "mlipkk_basis_function.h"

#include <cmath>

namespace MLIP_NS {

double cosine_cutoff_function(const double& dis, const double& cutoff) {
    if (dis < cutoff) {
        return 0.5 * (cos(M_PI * dis / cutoff) + 1.0);
    } else {
        return 0.0;
    }
}

double cosine_cutoff_function_d(const double& dis, const double& cutoff) {
    if (dis < cutoff) {
        return -0.5 * M_PI / cutoff * sin(M_PI * dis / cutoff);
    } else {
        return 0.0;
    }
}

double gauss(const double& dis, const double& beta, const double& mu) {
    return exp(-beta * (dis - mu) * (dis - mu));
}

/// @param[out] bf guassian
/// @param[out] bf_d derivative of gaussian w.r.t. distance
void gauss_d(const double& dis, const double& beta, const double& mu,
             double& bf, double& bf_d) {
    bf = gauss(dis, beta, mu);
    bf_d = -2.0 * beta * (dis - mu) * bf;
}

}  // namespace MLIP_NS
