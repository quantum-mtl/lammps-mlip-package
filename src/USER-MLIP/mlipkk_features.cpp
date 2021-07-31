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

#include "mlipkk_features.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <cstring>
#include <cassert>

#include "mlipkk_basis_function.h"
#include "mlipkk_spherical_harmonics.h"
#include "mlipkk_utils.h"

namespace MLIP_NS {

/// @param[out] fn fn[n] is value of radial functions with the n-th parameter
void get_fn(const double& dis, const double& cutoff, const char* radial_type,
            const vector2d& params, vector1d& fn)
{
    // TODO: other radial functions
    char accepted_radial_type[] = "gaussian";
    assert(strcmp(radial_type, accepted_radial_type) == 0);

    double fc = cosine_cutoff_function(dis, cutoff);

    fn.resize(params.size());
    for (int n = 0; n < static_cast<int>(params.size()); ++n){
        fn[n] = gauss(dis, params[n][0], params[n][1]) * fc;
    }
}


/// @param[out] fn fn[n] is value of radial functions with the n-th parameter
/// @param[out] fn_dr[n] is derivative of fn[n] w.r.t. radius
void get_fn(const double& dis, const double& cutoff, const char* radial_type,
            const vector2d& params, vector1d& fn, vector1d& fn_dr)
{
    // TODO: other radial functions
    char accepted_radial_type[] = "gaussian";
    assert(strcmp(radial_type, accepted_radial_type) == 0);

    const double fc = cosine_cutoff_function(dis, cutoff);
    const double fc_dr = cosine_cutoff_function_d(dis, cutoff);

    fn.resize(params.size());
    fn_dr.resize(params.size());
    double fn_val, fn_dr_val;
    for (int n = 0; n < static_cast<int>(params.size()); ++n){
        gauss_d(dis, params[n][0], params[n][1], fn_val, fn_dr_val);
        fn[n] = fn_val * fc;
        fn_dr[n] = fn_dr_val * fc + fn_val * fc_dr;
    }
}

/// @param[out] fn_ylm_dx (n_fn, n_lm_half)
/// @param[out] fn_ylm_dy (n_fn, n_lm_half)
/// @param[out] fn_ylm_dz (n_fn, n_lm_half)
void get_fn_ylm_dev(const double delx, const double dely, const double delz,
                    const SphericalHarmonics& sph,
                    const vector1d& fn, const vector1d& fn_d,
                    vector2dc& fn_ylm_dx, vector2dc& fn_ylm_dy, vector2dc& fn_ylm_dz)
{
    const auto r_polar_azimuthal = to_polar_coordinates(vector1d{delx, dely, delz});
    const double invdis = 1.0 / r_polar_azimuthal[0];
    const double costheta = cos(r_polar_azimuthal[1]);

    // derivative of spherical harmonics in cartesian coords.
    vector1dc ylm, ylm_dx, ylm_dy, ylm_dz;
    sph.compute_ylm(costheta, r_polar_azimuthal[2], ylm);
    sph.compute_ylm_der(costheta, r_polar_azimuthal[2], r_polar_azimuthal[0],
                        ylm_dx, ylm_dy, ylm_dz);

    const int n_fn = static_cast<int>(fn.size());
    const int n_lm_half = sph.get_n_lm_half();
    fn_ylm_dx = vector2dc(n_fn, vector1dc(n_lm_half));
    fn_ylm_dy = vector2dc(n_fn, vector1dc(n_lm_half));
    fn_ylm_dz = vector2dc(n_fn, vector1dc(n_lm_half));

    const double delx_invdis = delx * invdis;
    const double dely_invdis = dely * invdis;
    const double delz_invdis = delz * invdis;

    for (int n = 0; n < n_fn; ++n) {
        for (LMInfoIdx lm = 0; lm < n_lm_half; ++lm) {
            const dc f1 = fn_d[n] * ylm[lm];
            fn_ylm_dx[n][lm] = f1 * delx_invdis + fn[n] * ylm_dx[lm];
            fn_ylm_dy[n][lm] = f1 * dely_invdis + fn[n] * ylm_dy[lm];
            fn_ylm_dz[n][lm] = f1 * delz_invdis + fn[n] * ylm_dz[lm];
        }
    }
}

} // namespace MLIP_NS
