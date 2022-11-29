#ifndef MLIPKK_SPHERICAL_HARMONICS_KK_H_
#define MLIPKK_SPHERICAL_HARMONICS_KK_H_

#include "mlipkk_types.h"
#include "mlipkk_types_kokkos.h"

namespace MLIP_NS {

/* utility for spherical harmonics */
KOKKOS_INLINE_FUNCTION
int lm2i(int l, int m) { return (((l) * ((l) + 1)) / 2 + (m)); };

KOKKOS_INLINE_FUNCTION
Kokkos::complex<double> conj(Kokkos::complex<double> z) {
    return Kokkos::complex<double>(z.real(), -z.imag());
}

/// Update alp_kk_(npidx, :)
/// normalized associated Legendre polynomial P_{l}^{m}
/// For m >= 0, Y_{l}^{m} = P_{l}^{m} exp^{im phi} / sqrt(2)
KOKKOS_INLINE_FUNCTION
void compute_alp(const NeighborPairIdx npidx, const double costheta, int maxl,
                 const view_1d d_sph_coeffs_A, const view_1d d_sph_coeffs_B,
                 view_2d d_alp) {
    const double sintheta = sqrt(1.0 - costheta * costheta);

    double tmp = 0.39894228040143267794;  // = sqrt(0.5 / M_PI)
    d_alp(npidx, lm2i(0, 0)) = tmp;
    if (maxl == 0) {
        return;
    }

    const double SQRT3 = 1.7320508075688772935;
    d_alp(npidx, lm2i(1, 0)) = costheta * SQRT3 * tmp;
    const double SQRT3DIV2 = -1.2247448713915890491;
    tmp *= SQRT3DIV2 * sintheta;
    d_alp(npidx, lm2i(1, 1)) = tmp;

    for (int l = 2; l <= maxl; ++l) {
        for (int m = 0; m <= l - 2; ++m) {
            // DLMF 14.10.3
            d_alp(npidx, lm2i(l, m)) =
                d_sph_coeffs_A(lm2i(l, m)) *
                (costheta * d_alp(npidx, lm2i(l - 1, m)) +
                 d_sph_coeffs_B(lm2i(l, m)) * d_alp(npidx, lm2i(l - 2, m)));
        }
        d_alp(npidx, lm2i(l, l - 1)) =
            costheta * sqrt(2.0 * (l - 1.0) + 3.0) * tmp;
        tmp *= -sqrt(1.0 + 0.5 / l) * sintheta;
        // DLMF 14.7.15
        d_alp(npidx, lm2i(l, l)) = tmp;
    }
}

KOKKOS_INLINE_FUNCTION
void compute_ylm(const NeighborPairIdx npidx, const double azimuthal,
                 const int maxl, const view_2d d_alp, view_2dc d_ylm) {
    for (int l = 0; l <= maxl; ++l) {
        d_ylm(npidx, lm2i(l, l)) =
            d_alp(npidx, lm2i(l, 0)) * 0.5 * M_SQRT2;  // (l, m=0)
    }

    double c1 = 1.0, c2 = cos(azimuthal);   // cos(0 * phi) and cos(-1 * phi)
    double s1 = 0.0, s2 = -sin(azimuthal);  // sin(0 * phi) and sin(-1 * phi)
    const double tc = 2.0 * c2;
    double sign = -1;
    for (int mp = 1; mp <= maxl; ++mp) {
        const double s = tc * s1 - s2;  // sin(mp * phi)
        const double c = tc * c1 - c2;  // cos(mp * phi)
        c2 = c1;
        c1 = c;
        s2 = s1;
        s1 = s;
        for (int l = mp; l <= maxl; ++l) {
            const double tmp = sign * d_alp(npidx, lm2i(l, mp)) * 0.5 * M_SQRT2;
            d_ylm(npidx, lm2i(l, l - mp)) =
                tmp * Kokkos::complex<double>(c, -s);  // (l, m=-mp)
        }
        sign *= -1;
    }
}

/// normalized associated Legendre polynomial divied by sintheta,
/// P_{l}^{m}/sintheta
KOKKOS_INLINE_FUNCTION
void compute_alp_sintheta(const NeighborPairIdx npidx, const double costheta,
                          int maxl, const view_1d d_sph_coeffs_A,
                          const view_1d d_sph_coeffs_B,
                          view_2d d_alp_sintheta) {
    if (maxl == 0) {
        return;
    }

    double tmp = -0.48860251190291992263;  // -sqrt(3 / (4 * M_PI))

    d_alp_sintheta(npidx, lm2i(1, 1)) = tmp;

    const double sintheta = sqrt(1.0 - costheta * costheta);
    for (int l = 2; l <= maxl; ++l) {
        for (int m = 1; m <= l - 2; ++m) {
            // DLMF 14.10.3
            d_alp_sintheta(npidx, lm2i(l, m)) =
                d_sph_coeffs_A(lm2i(l, m)) *
                (costheta * d_alp_sintheta(npidx, lm2i(l - 1, m)) +
                 d_sph_coeffs_B(lm2i(l, m)) *
                     d_alp_sintheta(npidx, lm2i(l - 2, m)));
        }
        // DLMF
        d_alp_sintheta(npidx, lm2i(l, l - 1)) =
            costheta * sqrt(2.0 * (l - 1.0) + 3.0) * tmp;
        tmp *= -sqrt(1.0 + 0.5 / l) * sintheta;
        // DLMF 14.7.15
        d_alp_sintheta(npidx, lm2i(l, l)) = tmp;
    }
}

KOKKOS_INLINE_FUNCTION
void compute_ylm_der(const NeighborPairIdx npidx, const double costheta,
                     const double azimuthal, const double r, const int maxl,
                     const view_2d d_alp_sintheta, view_2dc d_ylm_dx,
                     view_2dc d_ylm_dy, view_2dc d_ylm_dz) {
    if (maxl == 0) {
        // We assume d_ylm_d[xyz] are filled by zero
        return;
    }

    const double sintheta = sqrt(1.0 - costheta * costheta);
    const double cosphi = cos(azimuthal);
    const double sinphi = sin(azimuthal);

    double c1 = 1.0, c2 = cosphi;   // cos(0 * phi) and cos(-1 * phi)
    double s1 = 0.0, s2 = -sinphi;  // sin(0 * phi) and sin(-1 * phi)
    const double tc = 2.0 * c2;
    const double invr = 1.0 / r;

    // (l, 0) for l >= 1
    for (int l = 1; l <= maxl; ++l) {
        const double common = d_alp_sintheta(npidx, lm2i(l, 1)) * sintheta *
                              invr * sqrt(0.5 * l * (l + 1));
        d_ylm_dx(npidx, lm2i(l, l)) = common * costheta * cosphi;
        d_ylm_dy(npidx, lm2i(l, l)) = common * costheta * sinphi;
        d_ylm_dz(npidx, lm2i(l, l)) = -common * sintheta;
    }

    double sign = -1.0;
    for (int mp = 1; mp <= maxl; ++mp) {
        const double s = tc * s1 - s2;  // sin(mp * phi)
        const double c = tc * c1 - c2;  // cos(mp * phi)
        c2 = c1;
        c1 = c;
        s2 = s1;
        s1 = s;
        for (int l = mp; l <= maxl; ++l) {
            const Kokkos::complex<double> eimphi(c, s);
            const auto common = eimphi * 0.5 * M_SQRT2 * invr;

            double dtheta = mp * costheta * d_alp_sintheta(npidx, lm2i(l, mp));
            if (mp != l) {
                dtheta += sqrt(static_cast<double>(l - mp) * (l + mp + 1)) *
                          d_alp_sintheta(npidx, lm2i(l, mp + 1)) *
                          sintheta;  // TODO: reuse p[]
            }
            const Kokkos::complex<double> dphi(
                0.0, mp * d_alp_sintheta(npidx, lm2i(l, mp)));

            d_ylm_dx(npidx, lm2i(l, l - mp)) =
                sign *
                conj(common * (dtheta * costheta * cosphi - dphi * sinphi));
            d_ylm_dy(npidx, lm2i(l, l - mp)) =
                sign *
                conj(common * (dtheta * costheta * sinphi + dphi * cosphi));
            d_ylm_dz(npidx, lm2i(l, l - mp)) =
                sign * conj(-common * dtheta * sintheta);
        }
        sign *= -1.0;
    }
}

}  // namespace MLIP_NS

#endif  // MLIPKK_SPHERICAL_HARMONICS_KK_H_
