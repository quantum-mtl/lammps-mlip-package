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

PairStyle(mlip_pair, PairMLIPPair)

#else

#ifndef LMP_PAIR_MLIP_PAIR_H
#define LMP_PAIR_MLIP_PAIR_H

#include "mlip_features.h"
#include "mlip_model.h"
#include "mlip_model_params.h"
#include "mlip_polynomial.h"
#include "mlip_polynomial_pair.h"
#include "mlip_pymlcpp.h"
#include "mlip_read_gtinv.h"
#include "pair.h"

namespace LAMMPS_NS {

class PairMLIPPair : public Pair {
   public:
    PairMLIPPair(class LAMMPS *);
    virtual ~PairMLIPPair();
    virtual void compute(int, int);
    void settings(int, char **);
    virtual void coeff(int, char **);

    virtual double init_one(int, int);
    /* virtual void init_style();
     */

   protected:
    virtual void allocate();
    vector1i types;

    MLIP_NS::DataMLIPBase<PolynomialPair> pot;

    const vector2i &get_type_comb() const;

    vector1d polynomial_model_uniq_products(const vector1d &dn);
    double dot(const vector1d &a, const vector1d &b, const int &sindex);

    void accumulate_energy_and_force_for_all_atom(int inum, int nlocal,
                                                  int newton_pair,
                                                  const vector2d &evdwl_array,
                                                  const vector2d &fpair_array);

    void compute_energy_and_force_for_each_atom(const vector3d &prod_all_f,
                                                const vector3d &prod_all_e,
                                                int ii, vector2d &evdwl_array,
                                                vector2d &fpair_array);

    void compute_partial_structural_feature_for_each_atom(const vector2d &dn,
                                                          int ii,
                                                          vector3d &prod_all_f,
                                                          vector3d &prod_all_e);

    void compute_main_structural_feature_for_each_atom(vector2d &dn, int ii);
};

}  // namespace LAMMPS_NS

#endif
#endif
