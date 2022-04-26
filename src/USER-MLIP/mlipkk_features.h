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

****************************************************************************/

#ifndef MLIPKK_FEATURES_H_
#define MLIPKK_FEATURES_H_

#include "mlipkk_spherical_harmonics.h"
#include "mlipkk_types.h"

namespace MLIP_NS {

// radial functions
void get_fn(const double& dis, const double& cutoff, const char* radial_type,
            const vector2d& params, vector1d& fn);
void get_fn(const double& dis, const double& cutoff, const char* radial_type,
            const vector2d& params, vector1d& fn, vector1d& fn_dr);

void get_fn_ylm_dev(const double delx, const double dely, const double delz,
                    const SphericalHarmonics& sph, const vector1d& fn,
                    const vector1d& fn_d, vector2dc& fn_ylm_dx,
                    vector2dc& fn_ylm_dy, vector2dc& fn_ylm_dz);

}  // namespace MLIP_NS

#endif  // MLIPKK_FEATURES_H_
