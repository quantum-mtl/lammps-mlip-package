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
#include "mlipkk_kokkos.h"
#include "mlipkk_types.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMLIPGtinvKokkos : public PairMLIPGtinv {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  PairMLIPGtinvKokkos(class LAMMPS *);
  virtual ~PairMLIPGtinvKokkos();

  void coeff(int, char**);
  void init_style();
  void compute(int, int);
  double init_one(int, int);
  void allocate();

 protected:
  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

//  DAT::tdual_efloat_1d k_eatom;
//  DAT::tdual_virial_array k_vatom;
//  typename AT::t_efloat_1d d_eatom;
//  typename AT::t_virial_array d_vatom;

  int inum;

  int eflag, vflag;

  int neighflag;

  Kokkos::View<T_INT*, DeviceType> d_map;                    // mapping from atom types to elements

  typedef Kokkos::DualView<F_FLOAT**, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
  typedef Kokkos::View<const F_FLOAT **, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
  t_fparams_rnd rnd_cutsq;
  
  // ---- for mlipkk ----
  std::vector<std::string> ele;
  typename MLIP_NS::vector1d mass;
  typename MLIP_NS::vector1d reg_coeffs;
  MLIP_NS::MLIPInput fp;
  typename MLIP_NS::Readgtinv gtinvdata;
  typename MLIP_NS::MLIPModel model;
  template<class PairStyle, class NeighListKokkos>
  friend void MLIP_NS::MLIPModel::set_structure_lmp(PairStyle *fpair, NeighListKokkos* k_list);
};
} // namespace LAMMPS

#endif //LMP_PAIR_MLIP_GTINV_KOKKOS_H_
#endif //PAIR_CLASS