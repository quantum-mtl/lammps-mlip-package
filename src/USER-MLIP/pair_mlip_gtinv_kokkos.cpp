//
// Created by Takayuki Nishiyama on 2021/04/01.
//

#include "pair_mlip_gtinv_kokkos.h"
#include "pair_mlip_gtinv_kokkos_impl.h"

namespace LAMMPS_NS{
template class PairMLIPGtinvKokkos<LMPDeviceType>;

#ifdef LMP_KOKKOS_GPU
template class PairMLIPGtinvKokkos<LMPHostType>;
#endif
}