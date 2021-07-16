//
// Created by Takayuki Nishiyama on 2021/07/16.
//

#ifndef LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_MLIPKK_LMP_H_
#define LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_MLIPKK_LMP_H_

#include "mlipkk.h"
#include "kokkos_type.h"

namespace MLIP_NS {
using LocalIdx = int;

class MLIPModelLMP : public MLIPModel {
 public:
  MLIPModelLMP() = default;
  ~MLIPModelLMP() = default;

  void initialize(const MLIPInput& input, const vector1d& reg_coeffs, const Readgtinv& gtinvdata);
  void compute();
  void compute_order_parameters();

  // defined here for LAMMPS interface
  template<class PairStyle, class NeighListKokkos>
  void set_structure(PairStyle *fpair, NeighListKokkos* k_list);

  // defined here for LAMMPS interface
  template<class PairStyle>
  void get_forces(PairStyle *fpair);

 protected:
  /* Total number of owned and ghost atoms on this proc*/
  int nall_;
};
}  // MLIP_NS
#endif //LAMMPS_MLIP_PACKAGE_SRC_USER_MLIP_MLIPKK_LMP_H_
