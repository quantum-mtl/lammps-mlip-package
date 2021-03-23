//
// Created by Takayuki Nishiyama on 2021/03/22.
//

#include "pair_mlip_pair_kokkos.h"
#include "pair_mlip_pair_kokkos_impl.h"

namespace LAMMPS_NS {
template class PairMLIPPairKokkos<LMPDeviceType>;

#ifdef KOKKOS_ENABLE_CUDA
template class PairMLIPPairKokkos<LMPHostType>;
#endif
}