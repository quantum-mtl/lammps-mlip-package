//
// Created by Takayuki Nishiyama on 2021/03/22.
//

#ifdef PAIR_CLASS

PairStyle(mlip_pair/kk,PairMLIPPairKokkos<LMPDeviceType>)
PairStyle(mlip_pair/kk/device,PairMLIPPairKokkos<LMPDeviceType>)
PairStyle(mlip_pair/kk/host,PairMLIPPairKokkos<LMPHostType>)

#else

#ifndef LMP_PAIR_MLIP_PAIR_KOKKOS_H
#define LMP_PAIR_MLIP_PAIR_KOKKOS_H

#include "kokkos_type.h"
#include "pair_kokkos.h"
#include "neigh_list_kokkos.h"

#include "pair_mlip_pair.h"
#include "mlip_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMLIPPairKokkos : public PairMLIPPair {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMLIPPairKokkos(class LAMMPS *);
  virtual ~PairMLIPPairKokkos();

  void init_style();
  void compute(int, int);
//
//  template<int NEIGHFLAG>
//  KOKKOS_INLINE_FUNCTION
//  void ev_tally(int i, int j, int nlocal, int newton_pair,
//                double evdwl, double ecoul, double fpair,
//                double delx, double dely, double delz)const;

 protected:
  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int inum;

  int eflag, vflag;

  int neighflag;

  typedef Kokkos::DualView<F_FLOAT **, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
  typedef Kokkos::View<const F_FLOAT **, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;

  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

};
}

#endif //LMP_PAIR_MLIP_PAIR_KOKKOS_H
#endif //PAIR_CLASS