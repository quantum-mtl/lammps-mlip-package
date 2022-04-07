/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Atsuto Seko
------------------------------------------------------------------------- */

#include "pair_mlip_gtinv.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairMLIPGtinv::PairMLIPGtinv(LAMMPS *lmp) : Pair(lmp) {
    single_enable = 0;
    restartinfo = 0;
    one_coeff = 1;
    manybody_flag = 1;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLIPGtinv::~PairMLIPGtinv() {
    if (copymode) return;

    if (allocated) {
        memory->destroy(setflag);
        memory->destroy(cutsq);
    }
}

/* ---------------------------------------------------------------------- */

void PairMLIPGtinv::compute(int eflag, int vflag) {
    vflag = 1;
    if (eflag || vflag) {
        ev_setup(eflag, vflag);
    } else {
        evflag = 0;
    }

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    const int n_type_comb = pot.get_model_params()
                                .get_type_comb_pair()
                                .size();  // equal to n_type * (n_type + 1) / 2
    const int n_fn = pot.get_model_params().get_n_fn();
    const int n_lm = pot.get_lm_info().size();
    const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;

    // order paramter, (inum, n_type, n_fn, n_lm_all)
    const barray4dc &anlm = compute_anlm();

    // partial a_nlm products
    barray4dc prod_anlm_f(boost::extents[n_type_comb][inum][n_fn][n_lm_all]);
    barray4dc prod_anlm_e(boost::extents[n_type_comb][inum][n_fn][n_lm_all]);
#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
        compute_partial_anlm_product_for_each_atom(n_fn, n_lm_all, anlm, ii,
                                                   prod_anlm_f, prod_anlm_e);
    }

    vector2d evdwl_array(inum), fx_array(inum), fy_array(inum), fz_array(inum);
    for (int ii = 0; ii < inum; ii++) {
        int i = list->ilist[ii];
        int jnum = list->numneigh[i];
        evdwl_array[ii].resize(jnum);
        fx_array[ii].resize(jnum);
        fy_array[ii].resize(jnum);
        fz_array[ii].resize(jnum);
    }

    vector1d scales;
    for (int l = 0; l < pot.get_feature_params().maxl + 1; ++l) {
        if (l % 2 == 0)
            for (int m = -l; m < l + 1; ++m) scales.emplace_back(1.0);
        else
            for (int m = -l; m < l + 1; ++m) scales.emplace_back(-1.0);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
        compute_energy_and_force_for_each_atom(prod_anlm_f, prod_anlm_e, scales,
                                               ii, evdwl_array, fx_array,
                                               fy_array, fz_array);
    }

    accumulate_energy_and_force_for_all_atom(
        inum, nlocal, newton_pair, evdwl_array, fx_array, fy_array, fz_array);
}

void PairMLIPGtinv::accumulate_energy_and_force_for_all_atom(
    int inum, int nlocal, int newton_pair, const vector2d &evdwl_array,
    const vector2d &fx_array, const vector2d &fy_array,
    const vector2d &fz_array) {
    int i, j, jnum, *jlist;
    double fx, fy, fz, evdwl, dis, delx, dely, delz;
    double **f = atom->f;
    double **x = atom->x;
    for (int ii = 0; ii < inum; ii++) {
        i = list->ilist[ii];
        jnum = list->numneigh[i], jlist = list->firstneigh[i];
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            delx = x[i][0] - x[j][0];
            dely = x[i][1] - x[j][1];
            delz = x[i][2] - x[j][2];
            dis = sqrt(delx * delx + dely * dely + delz * delz);
            if (dis < pot.get_feature_params().cutoff) {
                evdwl = evdwl_array[ii][jj];
                fx = fx_array[ii][jj];
                fy = fy_array[ii][jj];
                fz = fz_array[ii][jj];
                f[i][0] += fx, f[i][1] += fy, f[i][2] += fz;
                // if (newton_pair || j < nlocal)
                f[j][0] -= fx, f[j][1] -= fy, f[j][2] -= fz;
                if (evflag) {
                    ev_tally_xyz(i, j, nlocal, newton_pair, evdwl, 0.0, fx, fy,
                                 fz, delx, dely, delz);
                }
            }
        }
    }
}

void PairMLIPGtinv::compute_energy_and_force_for_each_atom(
    const barray4dc &prod_anlm_f, const barray4dc &prod_anlm_e,
    const vector1d &scales, int ii, vector2d &evdwl_array, vector2d &fx_array,
    vector2d &fy_array, vector2d &fz_array) {
    int i, j, jnum, *jlist, type1, type2, tc, m, lm1, lm2;
    double delx, dely, delz, dis, evdwl, fx, fy, fz, costheta, sintheta, cosphi,
        sinphi, coeff, cc;
    dc f1, ylm_dphi, d0, d1, d2, term1, term2, sume, sumf;

    double **x = atom->x;
    tagint *tag = atom->tag;

    i = list->ilist[ii];
    type1 = types[tag[i] - 1];
    jnum = list->numneigh[i];
    jlist = list->firstneigh[i];

    const int n_fn = pot.get_model_params().get_n_fn();
    const int n_des = pot.get_model_params().get_n_des();
    const int n_lm = pot.get_lm_info().size();
    const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;
    const int n_gtinv = pot.get_model_params().get_linear_term_gtinv().size();

    vector1d fn, fn_d;
    vector1dc ylm, ylm_dtheta;
    vector2dc fn_ylm, fn_ylm_dx, fn_ylm_dy, fn_ylm_dz;

    fn_ylm = fn_ylm_dx = fn_ylm_dy = fn_ylm_dz =
        vector2dc(n_fn, vector1dc(n_lm_all));

    for (int jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        delx = x[i][0] - x[j][0];
        dely = x[i][1] - x[j][1];
        delz = x[i][2] - x[j][2];
        dis = sqrt(delx * delx + dely * dely + delz * delz);

        if (dis < pot.get_feature_params().cutoff) {
            type2 = types[tag[j] - 1];

            const vector1d &sph =
                cartesian_to_spherical(vector1d{delx, dely, delz});
            get_fn(dis, pot.get_feature_params(), fn, fn_d);
            get_ylm(sph, pot.get_lm_info(), ylm, ylm_dtheta);

            costheta = cos(sph[0]), sintheta = sin(sph[0]);
            cosphi = cos(sph[1]), sinphi = sin(sph[1]);
            fabs(sintheta) > 1e-15 ? (coeff = 1.0 / sintheta) : (coeff = 0);
            for (int lm = 0; lm < n_lm; ++lm) {
                m = pot.get_lm_info()[lm][1], lm1 = pot.get_lm_info()[lm][2],
                lm2 = pot.get_lm_info()[lm][3];
                cc = pow(-1, m);
                ylm_dphi = dc{0.0, 1.0} * double(m) * ylm[lm];
                term1 = ylm_dtheta[lm] * costheta;
                term2 = coeff * ylm_dphi;
                d0 = term1 * cosphi - term2 * sinphi;
                d1 = term1 * sinphi + term2 * cosphi;
                d2 = -ylm_dtheta[lm] * sintheta;
                for (int n = 0; n < n_fn; ++n) {
                    fn_ylm[n][lm1] = fn[n] * ylm[lm];
                    fn_ylm[n][lm2] = cc * std::conj(fn_ylm[n][lm1]);
                    f1 = fn_d[n] * ylm[lm];
                    fn_ylm_dx[n][lm1] = -(f1 * delx + fn[n] * d0) / dis;
                    fn_ylm_dx[n][lm2] = cc * std::conj(fn_ylm_dx[n][lm1]);
                    fn_ylm_dy[n][lm1] = -(f1 * dely + fn[n] * d1) / dis;
                    fn_ylm_dy[n][lm2] = cc * std::conj(fn_ylm_dy[n][lm1]);
                    fn_ylm_dz[n][lm1] = -(f1 * delz + fn[n] * d2) / dis;
                    fn_ylm_dz[n][lm2] = cc * std::conj(fn_ylm_dz[n][lm1]);
                }
            }

            const int tc0 = get_type_comb()[type1][type2];
            const auto &prodif = prod_anlm_f[tc0][tag[i] - 1];
            const auto &prodie = prod_anlm_e[tc0][tag[i] - 1];
            const auto &prodjf = prod_anlm_f[tc0][tag[j] - 1];
            const auto &prodje = prod_anlm_e[tc0][tag[j] - 1];

            evdwl = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
            // including polynomial correction
            for (int n = 0; n < n_fn; ++n) {
                for (int lm0 = 0; lm0 < n_lm_all; ++lm0) {
                    sume = prodie[n][lm0] + prodje[n][lm0] * scales[lm0];
                    sumf = prodif[n][lm0] + prodjf[n][lm0] * scales[lm0];
                    evdwl += prod_real(fn_ylm[n][lm0], sume);
                    fx += prod_real(fn_ylm_dx[n][lm0], sumf);
                    fy += prod_real(fn_ylm_dy[n][lm0], sumf);
                    fz += prod_real(fn_ylm_dz[n][lm0], sumf);
                }
            }
            evdwl_array[ii][jj] = evdwl;
            fx_array[ii][jj] = fx;
            fy_array[ii][jj] = fy;
            fz_array[ii][jj] = fz;
        }
    }
}

/// @param[out] prod_anlm_f
/// @param[out] prod_anlm_e
void PairMLIPGtinv::compute_partial_anlm_product_for_each_atom(
    const int n_fn, const int n_lm_all, const barray4dc &anlm, int ii,
    barray4dc &prod_anlm_f, barray4dc &prod_anlm_e) {
    tagint *tag = atom->tag;
    int i = list->ilist[ii];
    int type1 = types[tag[i] - 1];

    // precompute partial a_nlm product for linear terms
    const int n_gtinv = pot.get_model_params().get_linear_term_gtinv().size();
    const vector2dc &uniq = compute_anlm_uniq_products(type1, anlm[tag[i] - 1]);

    // precompute partial a_nlm product for polynomial terms
    vector1d uniq_p;
    if (pot.get_feature_params().maxp > 1) {
        uniq_p = compute_polynomial_model_uniq_products(type1, anlm[tag[i] - 1],
                                                        uniq);
    }

    // compute prod_anlm_f and prod_anlm_e
    for (int type2 = 0; type2 < pot.get_feature_params().n_type; ++type2) {
        const int tc0 = get_type_comb()[type1][type2];

        for (int n = 0; n < n_fn; ++n) {
            for (int lm0 = 0; lm0 < n_lm_all; ++lm0) {
                dc sumf(0.0), sume(0.0);
                compute_anlm_linear_term(n_gtinv, tc0, n, lm0, uniq, sumf,
                                         sume);
                // polynomial model correction for sumf and sume
                compute_anlm_polynomial_model_correction(uniq_p, tc0, n, lm0,
                                                         uniq, sumf, sume);

                prod_anlm_f[tc0][tag[i] - 1][n][lm0] = sumf;
                prod_anlm_e[tc0][tag[i] - 1][n][lm0] = sume;
            }
        }
    }
}

/// @param[out] sumf
/// @param[out] sume
void PairMLIPGtinv::compute_anlm_linear_term(const int n_gtinv, const int tc0,
                                             int n, int lm0,
                                             const vector2dc &uniq, dc &sumf,
                                             dc &sume) {
    for (auto &inv : pot.get_poly_feature().get_gtinv_info(tc0, lm0)) {
        double regc = 0.5 * pot.get_reg_coeffs()[n * n_gtinv + inv.reg_i];
        if (inv.lmt_pi != -1) {
            dc valtmp = regc * inv.coeff * uniq[n][inv.lmt_pi];
            double valreal = valtmp.real() / inv.order;
            double valimag = valtmp.imag() / inv.order;
            sumf += valtmp;
            sume += dc(valreal, valimag);
        } else {
            sumf += regc;
            sume += regc;
        }
    }
}

/// @param[out] sumf
/// @param[out] sume
void PairMLIPGtinv::compute_anlm_polynomial_model_correction(
    const vector1d &uniq_p, const int tc0, int n, int lm0,
    const vector2dc &uniq, dc &sumf, dc &sume) {
    if (pot.get_feature_params().maxp <= 1) {
        return;
    }

    for (const auto &pi : pot.get_poly_feature().get_polynomial_info(
             tc0, n, lm0)) {  // The number of executions of this loop can be
                              // extremely large.
        double regc = pot.get_reg_coeffs()[pi.reg_i] * uniq_p[pi.comb_i];
        if (pi.lmt_pi != -1) {
            dc valtmp = regc * pi.coeff * uniq[n][pi.lmt_pi];
            double valreal = valtmp.real() / pi.order;
            double valimag = valtmp.imag() / pi.order;
            sumf += valtmp;
            sume += dc(valreal, valimag);
        } else {
            sumf += regc;
            sume += regc / pi.order;
        }
    }
}

/// @brief compute order parameters a_{nlm}
/// @return anlm (inum, n_type, n_fn, n_lm_all)
///         n_type is # of elements. n_fn is # of basis functions.
///         n_lm_all is # of (l, m), equal to (l_max + 1)^2
barray4dc PairMLIPGtinv::compute_anlm() {
    const int n_fn = pot.get_model_params().get_n_fn();
    const int n_lm = pot.get_lm_info().size();
    const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;
    const int n_type = pot.get_feature_params().n_type;

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    barray4dc anlm(boost::extents[inum][n_type][n_fn][n_lm_all]);
    barray4d anlm_r(boost::extents[inum][n_type][n_fn][n_lm]);
    barray4d anlm_i(boost::extents[inum][n_type][n_fn][n_lm]);
    std::fill(anlm_r.data(), anlm_r.data() + anlm_r.num_elements(), 0.0);
    std::fill(anlm_i.data(), anlm_i.data() + anlm_i.num_elements(), 0.0);

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
        int i, j, type1, type2, jnum, sindex, *ilist, *jlist;
        double delx, dely, delz, dis, scale;

        double **x = atom->x;
        tagint *tag = atom->tag;

        i = list->ilist[ii];
        type1 = types[tag[i] - 1];
        jnum = list->numneigh[i];
        jlist = list->firstneigh[i];

        vector1d fn;
        vector1dc ylm;
        dc val;
        for (int jj = 0; jj < jnum; ++jj) {
            j = jlist[jj];
            delx = x[i][0] - x[j][0];
            dely = x[i][1] - x[j][1];
            delz = x[i][2] - x[j][2];
            dis = sqrt(delx * delx + dely * dely + delz * delz);
            if (dis < pot.get_feature_params().cutoff) {
                type2 = types[tag[j] - 1];
                const vector1d &sph =
                    cartesian_to_spherical(vector1d{delx, dely, delz});
                get_fn(dis, pot.get_feature_params(), fn);
                get_ylm(sph, pot.get_lm_info(), ylm);
                for (int n = 0; n < n_fn; ++n) {
                    for (int lm = 0; lm < n_lm; ++lm) {
                        if (pot.get_lm_info()[lm][0] % 2 == 0)
                            scale = 1.0;
                        else
                            scale = -1.0;
                        val = fn[n] * ylm[lm];
#ifdef _OPENMP
#pragma omp atomic
#endif
                        anlm_r[tag[i] - 1][type2][n][lm] += val.real();
#ifdef _OPENMP
#pragma omp atomic
#endif
                        anlm_r[tag[j] - 1][type1][n][lm] += val.real() * scale;
#ifdef _OPENMP
#pragma omp atomic
#endif
                        anlm_i[tag[i] - 1][type2][n][lm] += val.imag();
#ifdef _OPENMP
#pragma omp atomic
#endif
                        anlm_i[tag[j] - 1][type1][n][lm] += val.imag() * scale;
                    }
                }
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(guided)
#endif
    for (int ii = 0; ii < inum; ii++) {
        int m, lm1, lm2;
        double cc;
        for (int type2 = 0; type2 < n_type; ++type2) {
            for (int n = 0; n < n_fn; ++n) {
                for (int lm = 0; lm < n_lm; ++lm) {
                    m = pot.get_lm_info()[lm][1];
                    lm1 = pot.get_lm_info()[lm][2],
                    lm2 = pot.get_lm_info()[lm][3];
                    anlm[ii][type2][n][lm1] = {anlm_r[ii][type2][n][lm],
                                               anlm_i[ii][type2][n][lm]};
                    cc = pow(-1, m);
                    anlm[ii][type2][n][lm2] = {cc * anlm_r[ii][type2][n][lm],
                                               -cc * anlm_i[ii][type2][n][lm]};
                }
            }
        }
    }

    return anlm;
}

/// @return prod vector2dc(n_fn, vector1dc: # of uniq products for
/// anlm_i[][n][])
vector2dc PairMLIPGtinv::compute_anlm_uniq_products(const int &type1,
                                                    const barray3dc &anlm_i) {
    const int n_fn = pot.get_model_params().get_n_fn();
    const vector3i &type_comb_pair =
        pot.get_model_params().get_type_comb_pair();
    const vector2i &uniq_prod = pot.get_poly_feature().get_uniq_prod();
    const vector2i &lmtc_map = pot.get_poly_feature().get_lmtc_map();

    vector2dc prod(n_fn, vector1dc(uniq_prod.size(), 1.0));
    for (int i = 0; i < uniq_prod.size(); ++i) {
        for (const auto &seq : uniq_prod[i]) {
            int lm = lmtc_map[seq][0];
            int tc = lmtc_map[seq][1];
            if (type_comb_pair[tc][type1].size() > 0) {
                int type2 = type_comb_pair[tc][type1][0];
                for (int n = 0; n < n_fn; ++n) {
                    prod[n][i] *= anlm_i[type2][n][lm];
                }
            } else {
                for (int n = 0; n < n_fn; ++n) {
                    prod[n][i] = 0.0;
                }
                break;
            }
        }
    }
    return prod;
}

/// @return uniq_prod vector<double>(uniq_comb.size())
vector1d PairMLIPGtinv::compute_polynomial_model_uniq_products(
    const int &type1, const barray3dc &anlm_i, const vector2dc &uniq) {
    const int n_fn = pot.get_model_params().get_n_fn();
    const int n_des = pot.get_model_params().get_n_des();
    const int n_gtinv = pot.get_model_params().get_linear_term_gtinv().size();
    const int n_lm = pot.get_lm_info().size();
    const int n_lm_all = 2 * n_lm - pot.get_feature_params().maxl - 1;
    const int n_type = pot.get_feature_params().n_type;

    vector1d dn = vector1d(n_des, 0.0);
    for (int type2 = 0; type2 < n_type; ++type2) {
        const int tc0 = get_type_comb()[type1][type2];
        for (int lm0 = 0; lm0 < n_lm_all; ++lm0) {
            for (const auto &t :
                 pot.get_poly_feature().get_gtinv_info_poly(tc0, lm0)) {
                if (t.lmt_pi == -1) {
                    for (int n = 0; n < n_fn; ++n) {
                        dn[n * n_gtinv + t.reg_i] += anlm_i[type2][n][0].real();
                    }
                } else {
                    for (int n = 0; n < n_fn; ++n) {
                        dn[n * n_gtinv + t.reg_i] +=
                            t.coeff / t.order *
                            prod_real(anlm_i[type2][n][lm0], uniq[n][t.lmt_pi]);
                    }
                }
            }
        }
    }

    const auto &uniq_comb = pot.get_poly_feature().get_uniq_comb();
    vector1d uniq_prod(uniq_comb.size(),
                       0.5);  // TODO: why initialize with 0.5?
    for (int n = 0; n < uniq_comb.size(); ++n) {
        for (const auto &c : uniq_comb[n]) {
            uniq_prod[n] *= dn[c];
        }
    }

    return uniq_prod;
}

/// @brief real part of complex products of val1 and val2
double PairMLIPGtinv::prod_real(const dc &val1, const dc &val2) {
    return val1.real() * val2.real() - val1.imag() * val2.imag();
}

/* ---------------------------------------------------------------------- */

void PairMLIPGtinv::allocate() {
    allocated = 1;
    int n = atom->ntypes;

    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; i++)
        for (int j = i; j <= n; j++) setflag[i][j] = 0;

    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLIPGtinv::settings(int narg, char **arg) {
    // force->newton_pair = 0;
    force->newton_pair = 1;
    if (narg != 0) error->all(FLERR, "Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLIPGtinv::coeff(int narg, char **arg) {
    if (!allocated) allocate();

    if (narg != 3 + atom->ntypes)
        error->all(FLERR, "Incorrect args for pair coefficients");

    // insure I,J args are * *
    if (strcmp(arg[0], "*") != 0 || strcmp(arg[1], "*") != 0)
        error->all(FLERR, "Incorrect args for pair coefficients");

    // read parameter file
    pot.initialize(arg[2]);
    if (pot.get_feature_params().des_type != "gtinv") {
        error->all(FLERR, "des_type must be gtinv");
    }

    // read args that map atom types to elements in potential file
    // map[i] = which element the Ith atom type is, -1 if NULL
    std::vector<int> map(atom->ntypes);
    const auto &ele = pot.get_elements();
    for (int i = 3; i < narg; i++) {
        for (int j = 0; j < ele.size(); j++) {
            if (strcmp(arg[i], ele[j].c_str()) == 0) {
                map[i - 3] = j;
                break;
            }
        }
    }

    const auto &mass = pot.get_masses();
    for (int i = 1; i <= atom->ntypes; ++i) {
        atom->set_mass(FLERR, i, mass[map[i - 1]]);
        for (int j = 1; j <= atom->ntypes; ++j) setflag[i][j] = 1;
    }

    for (int i = 0; i < atom->natoms; ++i) {
        types.emplace_back(map[(atom->type)[i] - 1]);
    }
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMLIPGtinv::init_one(int i, int j) {
    if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");

    return pot.get_cutmax();
}

/* ---------------------------------------------------------------------- */
const vector2i &PairMLIPGtinv::get_type_comb() const {
    return pot.get_type_comb();
}
