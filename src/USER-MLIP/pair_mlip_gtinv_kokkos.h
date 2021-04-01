//
// Created by Takayuki Nishiyama on 2021/04/01.
//

#ifdef PAIR_CLASS

PairStyle(mlip_gtinv/kk,PairMLIPGtinvKokkos<LMPDeviceType>)
PairStyle(mlip_gtinv/kk/device,PairMLIPGtinvKokkos<LMPDeviceType>)
PairStyle(mlip_gtinv/kk/host,PairMLIPGtinvKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_MLIP_GTINV_KOKKOS_H_
#define LMP_PAIR_MLIP_GTINV_KOKKOS_H_

#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

#include "pair_mlip_gtinv.h"
#include "mlip_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMLIPGtinvKokkos : public PairMLIPGtinv {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMLIPGtinvKokkos(class LAMMPS *);
  virtual ~PairMLIPGtinvKokkos();

  void init_style();
  void compute(int, int);

  void accumulate_energy_and_force_for_all_atom(int inum,
                                                int nlocal,
                                                int newton_pair,
                                                const vector2d &evdwl_array,
                                                const vector2d &fx_array,
                                                const vector2d &fy_array,
                                                const vector2d &fz_array);
  void compute_energy_and_force_for_each_atom(const barray4dc &prod_anlm_f,
                                              const barray4dc &prod_anlm_e,
                                              const vector1d &scales,
                                              int ii,
                                              vector2d &evdwl_array,
                                              vector2d &fx_array,
                                              vector2d &fy_array,
                                              vector2d &fz_array);
  void compute_partial_anlm_product_for_each_atom(const int n_fn,
                                                  const int n_lm_all,
                                                  const barray4dc &anlm,
                                                  int ii,
                                                  barray4dc &prod_anlm_f,
                                                  barray4dc &prod_anlm_e);
  barray4dc compute_anlm();

 protected:
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;
  HAT::t_neighbors_2d h_neighbors;
  HAT::t_int_1d_randomread h_ilist;
  HAT::t_int_1d_randomread h_numneigh;

//  DAT::tdual_efloat_1d k_eatom;
//  DAT::tdual_virial_array k_vatom;
//  typename AT::t_efloat_1d d_eatom;
//  typename AT::t_virial_array d_vatom;

//  int inum;

  int eflag, vflag;

  int neighflag;
};
}

#endif //LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_PAIR_MLIP_GTINV_KOKKOS_H_
#endif //PAIR_CLASS