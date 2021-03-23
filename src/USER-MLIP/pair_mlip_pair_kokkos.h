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
  ~PairMLIPPairKokkos();
 protected:
  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typedef Kokkos::DualView<F_FLOAT **, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
  typedef Kokkos::View<const F_FLOAT **, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;
};
}

#endif //LMP_PAIR_MLIP_PAIR_KOKKOS_H
#endif //PAIR_CLASS