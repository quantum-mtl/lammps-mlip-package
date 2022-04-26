#ifndef MLIPKK_FEATURES_KK_H_
#define MLIPKK_FEATURES_KK_H_

#include "mlipkk_types.h"
#include "mlipkk_types_kokkos.h"

namespace MLIP_NS {

KOKKOS_INLINE_FUNCTION
double cosine_cutoff_function_kk(const double dis, const double cutoff) {
    if (dis < cutoff) {
        return 0.5 * (cos(M_PI * dis / cutoff) + 1.0);
    } else {
        return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
double cosine_cutoff_function_d_kk(const double dis, const double cutoff) {
    if (dis < cutoff) {
        return -0.5 * M_PI / cutoff * sin(M_PI * dis / cutoff);
    } else {
        return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
double gauss_kk(const double dis, const double beta, const double mu) {
    return exp(-beta * (dis - mu) * (dis - mu));
}

// radial functions
KOKKOS_INLINE_FUNCTION
void get_fn_kk(const NeighborPairIdx npidx, const double r, const double cutoff,
               const view_2d d_params, view_2d d_fn, view_2d d_fn_der) {
    const double fc = cosine_cutoff_function_kk(r, cutoff);
    const double fc_dr = cosine_cutoff_function_d_kk(r, cutoff);

    const int num_params = d_params.extent(0);
    for (int n = 0; n < num_params; ++n) {
        // d_params(n, :) = [beta, mu]
        const double bf = gauss_kk(r, d_params(n, 0), d_params(n, 1));
        const double bf_der = -2.0 * d_params(n, 0) * (r - d_params(n, 1)) * bf;

        d_fn(npidx, n) = bf * fc;
        d_fn_der(npidx, n) = bf_der * fc + bf * fc_dr;
    }
}

}  // namespace MLIP_NS
#endif  // MLIPKK_FEATURES_KK_H_
