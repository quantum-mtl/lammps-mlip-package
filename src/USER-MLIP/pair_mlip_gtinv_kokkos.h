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
#include "mlipkk_lmp.h"
#include "mlipkk_types.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairMLIPGtinvKokkos : public PairMLIPGtinv {
 public:
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;

  explicit PairMLIPGtinvKokkos(class LAMMPS *);
  ~PairMLIPGtinvKokkos() override;

  void settings(int narg, char **arg) override;
  void coeff(int, char **) override;
  void init_style() override;
  void compute(int, int) override;
  double init_one(int, int) override;
  void allocate() override;

 protected:
  typename AT::t_x_array_randomread x;
  typename AT::t_f_array f;

  // From pair_eam_alloy_kokkos.h
  // Declare dups for HALFTHREAD
  // need_dup is HALFTHREAD(4u) when threads are gt 2 or gpus are gt 1.
  // https://github.com/lammps/lammps/blob/998b76520e74b3a90580bf1a92155dcbe2843dba/src/KOKKOS/pair_eam_alloy_kokkos.h#L130-L138
  int need_dup;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterDuplicated> dup_vatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_f;
  Kokkos::Experimental::ScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_eatom;
  Kokkos::Experimental::ScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,typename KKDevice<DeviceType>::value,typename Kokkos::Experimental::ScatterSum,Kokkos::Experimental::ScatterNonDuplicated> ndup_vatom;


  typename AT::t_neighbors_2d d_neighbors;
  typename AT::t_int_1d_randomread d_ilist;
  typename AT::t_int_1d_randomread d_numneigh;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  int inum;  // number of I atoms neighbors are stored for

  int eflag, vflag;

  int neighflag;
  int newton_pair;

//  Kokkos::View<T_INT *, DeviceType> d_map;                    // mapping from atom types to elements

  typedef Kokkos::DualView<F_FLOAT **, DeviceType> tdual_fparams;
  tdual_fparams k_cutsq;
//  typedef Kokkos::View<const F_FLOAT **, DeviceType, Kokkos::MemoryTraits<Kokkos::RandomAccess>> t_fparams_rnd;
//  t_fparams_rnd rnd_cutsq;

  // ---- for mlipkk ----
  std::vector<std::string> ele;
  std::vector<int> map;
  typename MLIP_NS::vector1d mass;
  typename MLIP_NS::vector1d reg_coeffs;
  typename MLIP_NS::MLIPInput *fp;
  typename MLIP_NS::Readgtinv gtinvdata;
  typename MLIP_NS::MLIPModelLMP *model;
  template<class PairStyle, class NeighListKokkos>
  friend void MLIP_NS::MLIPModelLMP::set_structure(PairStyle *fpair, NeighListKokkos *k_list);
  friend void pair_virial_fdotr_compute<PairMLIPGtinvKokkos>(PairMLIPGtinvKokkos *);
  template<class PairStyle>
  friend void MLIP_NS::MLIPModelLMP::get_forces(PairStyle *fpair);
};
} // namespace LAMMPS

#endif //LMP_PAIR_MLIP_GTINV_KOKKOS_H_
#endif //PAIR_CLASS