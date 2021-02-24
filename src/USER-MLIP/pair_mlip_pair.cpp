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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "atom.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "force.h"
#include "comm.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "memory.h"
#include "error.h"

#include "pair_mlip_pair.h"

using namespace LAMMPS_NS;

#define MAXLINE 1024
#define DELTA 4

/* ---------------------------------------------------------------------- */

PairMLIPPair::PairMLIPPair(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 0;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairMLIPPair::~PairMLIPPair()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

/* ---------------------------------------------------------------------- */

void PairMLIPPair::compute(int eflag, int vflag)
{

    vflag = 1;
    if (eflag || vflag) ev_setup(eflag,vflag);
    else evflag = 0;

    int inum = list->inum;
    int nlocal = atom->nlocal;
    int newton_pair = force->newton_pair;

    // first part of polynomial model correction
    const int n_type_comb = pot.get_model_params().get_type_comb_pair().size();
    vector3d prod_all_f(n_type_comb, vector2d(inum));
    vector3d prod_all_e(n_type_comb, vector2d(inum));
    if (pot.get_feature_params().maxp > 1){
        vector2d dn(inum, vector1d(pot.get_model_params().get_n_des(), 0.0));
        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided)
        #endif
        for (int ii = 0; ii < inum; ii++) {
            compute_main_structural_feature_for_each_atom(dn, ii);

        }

        #ifdef _OPENMP
        #pragma omp parallel for schedule(guided)
        #endif
        for (int ii = 0; ii < inum; ii++) {
            compute_partial_structural_feature_for_each_atom(dn, ii, prod_all_f, prod_all_e);
        }
    }
    // end: first part of polynomial model correction

    vector2d evdwl_array(inum),fpair_array(inum);
    #ifdef _OPENMP
    #pragma omp parallel for schedule(guided)
    #endif
    for (int ii = 0; ii < inum; ii++) {
        compute_energy_and_force_for_each_atom(prod_all_f, prod_all_e, ii, evdwl_array, fpair_array);
    }

    accumulate_energy_and_force_for_all_atom(inum, nlocal, newton_pair, evdwl_array, fpair_array);
}

void PairMLIPPair::compute_main_structural_feature_for_each_atom(vector2d &dn, int ii) {
    int i,j,type1,type2,jnum,sindex,*ilist,*jlist;
    double delx,dely,delz,dis;
    double **x = atom->x;
    tagint *tag = atom->tag;

    vector1d fn;
    const int n_fn = pot.get_model_params().get_n_fn();

    i = list->ilist[ii];
    type1 = types[tag[i] - 1];
    jnum = list->numneigh[i];
    jlist = list->firstneigh[i];
    for (int jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        delx = x[i][0]-x[j][0];
        dely = x[i][1]-x[j][1];
        delz = x[i][2]-x[j][2];
        dis = sqrt(delx*delx + dely*dely + delz*delz);

        if (dis < pot.get_feature_params().cutoff){
            type2 = types[tag[j] - 1];
            sindex = get_type_comb()[type1][type2] * n_fn;
            get_fn(dis, pot.get_feature_params(), fn);
            for (int n = 0; n < n_fn; ++n) {
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                dn[tag[i]-1][sindex+n] += fn[n];
                #ifdef _OPENMP
                #pragma omp atomic
                #endif
                dn[tag[j]-1][sindex+n] += fn[n];
            }
        }
    }
}

void PairMLIPPair::compute_partial_structural_feature_for_each_atom(const vector2d &dn, int ii, vector3d &prod_all_f, vector3d &prod_all_e) {
    int i,*ilist,type1;
    double **x = atom->x;
    tagint *tag = atom->tag;
    ilist = list->ilist;

    i = ilist[ii], type1 = types[tag[i] - 1];
    const int n_fn = pot.get_model_params().get_n_fn();
    const vector1d &prodi
        = polynomial_model_uniq_products(dn[tag[i] - 1]);
    for (int type2 = 0; type2 < pot.get_feature_params().n_type; ++type2){
        const int tc = get_type_comb()[type1][type2];
        vector1d vals_f(n_fn, 0.0), vals_e(n_fn, 0.0);
        for (int n = 0; n < n_fn; ++n){
            double v;
            for (const auto& pi:
                pot.get_poly_feature().get_polynomial_info(tc, n)){
                v = prodi[pi.comb_i] * pot.get_reg_coeffs()[pi.reg_i];
                vals_f[n] += v;
                vals_e[n] += v / pi.order;
            }
        }
        prod_all_f[tc][tag[i]-1] = vals_f;
        prod_all_e[tc][tag[i]-1] = vals_e;
    }
}

void
PairMLIPPair::compute_energy_and_force_for_each_atom(const vector3d &prod_all_f, const vector3d &prod_all_e, int ii,
                                                     vector2d &evdwl_array, vector2d &fpair_array) {
    int i,j,jnum,*jlist,type1,type2,sindex,tc;
    double delx,dely,delz,dis,fpair,evdwl;
    double **x = atom->x;
    tagint *tag = atom->tag;

    i = list->ilist[ii];
    type1 = types[tag[i] - 1];
    jnum = list->numneigh[i];
    jlist = list->firstneigh[i];

    const int n_fn = pot.get_model_params().get_n_fn();
    vector1d fn, fn_d;

    evdwl_array[ii].resize(jnum);
    fpair_array[ii].resize(jnum);
    for (int jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        delx = x[i][0]-x[j][0];
        dely = x[i][1]-x[j][1];
        delz = x[i][2]-x[j][2];
        dis = sqrt(delx*delx + dely*dely + delz*delz);

        if (dis < pot.get_feature_params().cutoff){
            type2 = types[tag[j] - 1];
            tc = get_type_comb()[type1][type2];
            sindex = tc * n_fn;

            get_fn(dis, pot.get_feature_params(), fn, fn_d);
            fpair = dot(fn_d, pot.get_reg_coeffs(), sindex);
            evdwl = dot(fn, pot.get_reg_coeffs(), sindex);

            // polynomial model correction
            if (pot.get_feature_params().maxp > 1){
                fpair += dot(fn_d, prod_all_f[tc][tag[i] - 1], 0)
                         + dot(fn_d, prod_all_f[tc][tag[j] - 1], 0);
                evdwl += dot(fn, prod_all_e[tc][tag[i] - 1], 0)
                         + dot(fn, prod_all_e[tc][tag[j] - 1], 0);
            }
            // polynomial model correction: end

            fpair *= - 1.0 / dis;
            evdwl_array[ii][jj] = evdwl;
            fpair_array[ii][jj] = fpair;
        }
    }
}

void PairMLIPPair::accumulate_energy_and_force_for_all_atom(int inum, int nlocal, int newton_pair,
                                                            const vector2d &evdwl_array,
                                                            const vector2d &fpair_array) {
    int i,j,jnum,*jlist;
    double fpair,evdwl,dis,delx,dely,delz;
    double **f = atom->f;
    double **x = atom->x;
    for (int ii = 0; ii < inum; ii++) {
        i = list->ilist[ii];
        jnum = list->numneigh[i], jlist = list->firstneigh[i];
        for (int jj = 0; jj < jnum; jj++) {
            j = jlist[jj];
            delx = x[i][0]-x[j][0];
            dely = x[i][1]-x[j][1];
            delz = x[i][2]-x[j][2];
            dis = sqrt(delx*delx + dely*dely + delz*delz);
            if (dis < pot.get_feature_params().cutoff){
                evdwl = evdwl_array[ii][jj];
                fpair = fpair_array[ii][jj];
                f[i][0] += fpair*delx;
                f[i][1] += fpair*dely;
                f[i][2] += fpair*delz;
                //            if (newton_pair || j < nlocal)
                f[j][0] -= fpair*delx;
                f[j][1] -= fpair*dely;
                f[j][2] -= fpair*delz;
                if (evflag) {
                    ev_tally(i, j, nlocal, newton_pair,
                             evdwl, 0.0, fpair, delx, dely, delz);
                }
            }
        }
    }
}

vector1d PairMLIPPair::polynomial_model_uniq_products(const vector1d& dn){

    const auto &uniq_comb = pot.get_poly_feature().get_uniq_comb();
    vector1d prod(uniq_comb.size(), 0.5);
    for (int n = 0; n < uniq_comb.size(); ++n){
        for (const auto& c: uniq_comb[n]) prod[n] *= dn[c];
    }

    return prod;
}

double PairMLIPPair::dot
(const vector1d& a, const vector1d& b, const int& sindex){
    double val(0.0);
    for (int n = 0; n < a.size(); ++n) val += a[n] * b[sindex+n];
    return val;
}

/* ---------------------------------------------------------------------- */

void PairMLIPPair::allocate()
{

  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
      for (int j = i; j <= n; j++)
      setflag[i][j] = 0;


  memory->create(setflag,n+1,n+1,"pair:setflag");
  memory->create(cutsq,n+1,n+1,"pair:cutsq");

}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairMLIPPair::settings(int narg, char **arg)
{
  force->newton_pair = 1;
  if (narg != 0) error->all(FLERR,"Illegal pair_style command");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairMLIPPair::coeff(int narg, char **arg)
{
    if (!allocated) allocate();

    if (narg != 3 + atom->ntypes)
        error->all(FLERR,"Incorrect args for pair coefficients");

    // insure I,J args are * *
    if (strcmp(arg[0],"*") != 0 || strcmp(arg[1],"*") != 0)
        error->all(FLERR,"Incorrect args for pair coefficients");

    pot.initialize(arg[2]);
    if (pot.get_feature_params().des_type != "pair"){
        error->all(FLERR,"des_type must be pair");
    }

    // read args that map atom types to elements in potential file
    // map[i] = which element the Ith atom type is, -1 if NULL
    std::vector<int> map(atom->ntypes);
    auto ele = pot.get_elements();
    for (int i = 3; i < narg; i++) {
        for (int j = 0; j < ele.size(); j++){
            if (strcmp(arg[i],ele[j].c_str()) == 0){
                map[i-3] = j;
                break;
            }
        }
    }

    auto mass = pot.get_masses();
    for (int i = 1; i <= atom->ntypes; ++i){
        atom->set_mass(FLERR,i,mass[map[i-1]]);
        for (int j = 1; j <= atom->ntypes; ++j) setflag[i][j] = 1;
    }

    for (int i = 0; i < atom->natoms; ++i){
        types.emplace_back(map[(atom->type)[i]-1]);
    }
}


/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairMLIPPair::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  return pot.get_cutmax();
}

/* ---------------------------------------------------------------------- */
const vector2i& PairMLIPPair::get_type_comb() const {
    return pot.get_type_comb();
}
