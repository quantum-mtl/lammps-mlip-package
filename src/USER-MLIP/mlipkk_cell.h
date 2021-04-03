#ifndef MLIP_CELL_H
#define MLIP_CELL_H

#include "mlipkk_types.h"

namespace MLIP_NS {

/// @param[out] displacements
/// @param[out] neighbors
void get_neighbors(const vector2d& coords, const vector2d& lattice, const double cutoff,
                   vector3d& displacements, vector2i& neighbors);

} // namespace MLIP_NS

#endif
