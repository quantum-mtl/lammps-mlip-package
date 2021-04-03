#include "mlipkk_cell.h"

#include <vector>
#include <cmath>
#include <iostream>

#include "mlipkk_types.h"

namespace MLIP_NS {

void get_neighbors(const vector2d& coords, const vector2d& lattice, const double cutoff,
                   vector3d& displacements, vector2i& neighbors)
{
    int inum = static_cast<int>(coords.size());
    displacements.resize(inum);
    neighbors.resize(inum);

    std::vector<int> bounds(3);
    for (int i = 0; i < 3; ++i) {
        double length = sqrt(lattice[i][0] * lattice[i][0] + lattice[i][1] * lattice[i][1] + lattice[i][2] * lattice[i][2]);
        bounds[i] = static_cast<int>(ceil(length / cutoff));
    }

    std::vector<std::vector<double>> offsets;
    for (int nx = -bounds[0]; nx <= bounds[0]; ++nx) {
        for (int ny = -bounds[1]; ny <= bounds[1]; ++ny) {
            for (int nz = -bounds[2]; nz <= bounds[2]; ++nz) {
                std::vector<double> offset(3, 0.0);
                for (int i = 0; i < 3; ++i) {
                    offset[i] += lattice[0][i] * nx + lattice[1][i] * ny + lattice[2][i] * nz;
                }
                offsets.emplace_back(offset);
            }
        }
    }

    double r2 = cutoff * cutoff;
    for (int i = 0; i < inum; ++i) {
        for (int j = i; j < inum; ++j) {
            std::vector<double> rij(3, 0.0);
            for (int x = 0; x < 3; ++x) {
                rij[x] = coords[j][x] - coords[i][x];
            }
            for (auto offset: offsets) {
                std::vector<double> disp(rij);
                for (int x = 0; x < 3; ++x) {
                    disp[x] += offset[x];
                }

                if ((fabs(disp[0]) > cutoff) || (fabs(disp[1]) > cutoff) || (fabs(disp[2]) > cutoff)) {
                    continue;
                }

                double length2 = disp[0] * disp[0] + disp[1] * disp[1] + disp[2] * disp[2];

                // same site
                if ((i == j) && (length2 < 1e-8)) {
                    continue;
                }

                if (length2 < r2) {
                    if (i == j) {
                        std::cerr << "Cutoff radius is too short!" << std::endl;
                    }
                    displacements[i].emplace_back(disp);
                    neighbors[i].emplace_back(j);
                }
            }
        }
    }
}

} // namespace MLIP_NS
