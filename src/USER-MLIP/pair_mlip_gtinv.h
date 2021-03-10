/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlip_gtinv,PairMLIPGtinv)

#else

#ifndef LMP_PAIR_MLIP_GTINV_H
#define LMP_PAIR_MLIP_GTINV_H

#include "pair.h"

#include "mlip_pymlcpp.h"
#include "mlip_features.h"
#include "mlip_model_params.h"
#include "mlip_model.h"

namespace LAMMPS_NS {

class PairMLIPGtinv : public Pair {
public:
    PairMLIPGtinv(class LAMMPS *);
    virtual ~PairMLIPGtinv();
    virtual void compute(int, int);
    void settings(int, char **);
    virtual void coeff(int, char **);

    virtual double init_one(int, int);
    // virtual void init_style();

protected:
    virtual void allocate();
    vector1i types;

    MLIP_NS::DataMLIPBase<PolynomialGtinv> pot;

    const vector2i& get_type_comb() const;

    barray4dc compute_anlm();

    vector2dc compute_anlm_uniq_products(const int& type1, const barray3dc& anlm_i);
    vector1d compute_polynomial_model_uniq_products(const int& type1,
                                                    const barray3dc& anlm_i,
                                                    const vector2dc& prod);
    // vector1d polynomial_model_uniq_products(const vector1d& dn);

    double prod_real(const dc& val1, const dc& val2);

    void compute_partial_anlm_product_for_each_atom(const int n_fn, const int n_lm_all,
                                                    const barray4dc &anlm, int ii,
                                                    barray4dc &prod_anlm_f,
                                                    barray4dc &prod_anlm_e);
    void update_partial_anlm_product_of_Iatom(const int n_fn,
                                              const int n_lm_all,
                                              barray4dc &prod_anlm_f,
                                              barray4dc &prod_anlm_e,
                                              int i,
                                              int type1,
                                              const tagint *tag,
                                              const int n_gtinv,
                                              const vector2dc &uniq,
                                              const vector1d &uniq_p);
    void compute_anlm_linear_term(const int n_gtinv, const int tc0, int n, int lm0,
                                  const vector2dc &uniq, dc &sumf, dc &sume);
    void compute_anlm_polynomial_model_correction(const vector1d &uniq_p,
                                                  const int tc0, int n, int lm0, const vector2dc &uniq,
                                                  dc &sumf, dc &sume);

    void
    compute_energy_and_force_for_each_atom(const barray4dc &prod_anlm_f, const barray4dc &prod_anlm_e, const vector1d &scales,
                                           int ii,
                                           vector2d &evdwl_array, vector2d &fx_array, vector2d &fy_array, vector2d &fz_array);

    void accumulate_energy_and_force_for_all_atom(int inum, int nlocal, int newton_pair, const vector2d &evdwl_array,
                                                  const vector2d &fx_array, const vector2d &fy_array, const vector2d &fz_array);

};

}

#endif
#endif
