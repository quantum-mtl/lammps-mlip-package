//
// Created on 2021/07/16.
//
#include "kokkos_type.h"
#include "mlipkk_lmp.h"

namespace MLIP_NS{

void MLIPModelLMP::initialize(const MLIPInput& input, const vector1d& reg_coeffs, const Readgtinv& gtinvdata)
{
  MLIPModel::initialize(input, reg_coeffs, gtinvdata);
  nall_ = 0;
}

} // namespace MLIP_NS
