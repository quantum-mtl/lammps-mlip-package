#include "mlipkk_cell.h"

#include <vector>
#include <cmath>
#include <iostream>
#include <cassert>

#include "mlipkk_types.h"

namespace MLIP_NS {

// ----------------------------------------------------------------------------
// NBin class
// ----------------------------------------------------------------------------

NBin::NBin(const std::vector<std::vector<double>>& lattice, const double cutoff) : OUTSIDE(-1)
{
    const double lx = norm(lattice[0][0], lattice[0][1], lattice[0][2]);
    const double ly = norm(lattice[1][0], lattice[1][1], lattice[1][2]);
    const double lz = norm(lattice[2][0], lattice[2][1], lattice[2][2]);
    const double binsize_optimal = 0.5 * cutoff;  // corresponds to BIN style

    // global
    const double EPS = 1e-8;
    nbinx_ = static_cast<int>(floor(lx * (1.0 - EPS) / binsize_optimal));
    nbiny_ = static_cast<int>(floor(ly * (1.0 - EPS) / binsize_optimal));
    nbinz_ = static_cast<int>(floor(lz * (1.0 - EPS) / binsize_optimal));
    binsizex_ = lx / nbinx_;
    binsizey_ = ly / nbiny_;
    binsizez_ = lz / nbinz_;
    assert(binsizex_ >= binsize_optimal);
    assert(binsizey_ >= binsize_optimal);
    assert(binsizez_ >= binsize_optimal);

    // local
    mbinxlo_ = static_cast<int>(floor((-cutoff - EPS * lx) / binsizex_));
    mbinylo_ = static_cast<int>(floor((-cutoff - EPS * ly) / binsizey_));
    mbinzlo_ = static_cast<int>(floor((-cutoff - EPS * lz) / binsizez_));

    mbinxhi_ = static_cast<int>(ceil((lx * (1.0 + EPS) + cutoff) / binsizex_));
    mbinyhi_ = static_cast<int>(ceil((ly * (1.0 + EPS) + cutoff) / binsizey_));
    mbinzhi_ = static_cast<int>(ceil((lz * (1.0 + EPS) + cutoff) / binsizez_));

    mbinx_ = mbinxhi_ - mbinxlo_ + 1;
    mbiny_ = mbinyhi_ - mbinylo_ + 1;
    mbinz_ = mbinzhi_ - mbinzlo_ + 1;

    mbins_ = mbinx_ * mbiny_ * mbinz_;
}

void NBin::bin_atoms(const std::vector<std::vector<double>>& coords_with_ghosts) {
    const int nall = static_cast<int>(coords_with_ghosts.size());

    bins_ = std::vector<std::vector<PeriodicSiteIdx>>(mbins_);
    atom2bin_ = std::vector<BinIdx>(nall);

    for (PeriodicSiteIdx i = 0; i < nall; ++i) {
        const BinIdx ibin = coords2bin(coords_with_ghosts[i]);
        atom2bin_[i] = ibin;
        if (ibin != OUTSIDE) {
            bins_[ibin].emplace_back(i);
        }
    }
}

BinIdx NBin::hash_bin(const int binx, const int biny, const int binz) const {
    return (binz - mbinzlo_) * mbiny_ * mbinx_ + (biny - mbinylo_) * mbinx_ + (binx - mbinxlo_);
};

BinIdx NBin::coords2bin(const std::vector<double>& xyz) const {
    const int binx = static_cast<int>(floor(xyz[0] / binsizex_));
    const int biny = static_cast<int>(floor(xyz[1] / binsizey_));
    const int binz = static_cast<int>(floor(xyz[2] / binsizez_));

    if ((binx >= mbinxlo_) && (binx <= mbinxhi_)
        && (biny >= mbinylo_) && (biny <= mbinyhi_)
        && (binz >= mbinzlo_) && (binz <= mbinzhi_)
    ) {
        return hash_bin(binx, biny, binz);
    } else {
        return OUTSIDE;
    }
}

std::vector<int> NBin::get_nbin_size() const {
    return std::vector<int>{nbinx_, nbiny_, nbinz_};
}

std::vector<int> NBin::get_mbinlo() const {
    return std::vector<int>{mbinxlo_, mbinylo_, mbinzlo_};
}

std::vector<int> NBin::get_mbinhi() const {
    return std::vector<int>{mbinxhi_, mbinyhi_, mbinzhi_};
}

int NBin::get_mbins() const {
    return mbins_;
}

const std::vector<PeriodicSiteIdx>& NBin::get_atoms_within_bin(const BinIdx ibin) const {
    return bins_[ibin];
}

BinIdx NBin::get_binidx(const PeriodicSiteIdx i) const {
    return atom2bin_[i];
}

void NBin::dump(std::ostream& os) const {
    os << "Bin sizes: " << binsizex_ << " " << binsizey_ << " " << binsizez_ << std::endl;
    os << "Global bins: " << nbinx_ << " " << nbiny_ << " " << nbinz_ << std::endl;
    os << "Lowest local bin : " << mbinxlo_ << " " << mbinylo_ << " " << mbinzlo_ << std::endl;
    os << "Highest local bin: " << mbinxhi_ << " " << mbinyhi_ << " " << mbinzhi_ << std::endl;
    os << "mbins: " << mbins_ << std::endl;
}

// ----------------------------------------------------------------------------
// NStencil class
// ----------------------------------------------------------------------------

NStencilHalf::NStencilHalf(const NBin& nbin) {
    const int mbins = nbin.get_mbins();
    stencils_.resize(mbins);

    const auto nbin_size = nbin.get_nbin_size();
    const auto nbin_lo = nbin.get_mbinlo();
    const auto nbin_hi = nbin.get_mbinhi();

    // assign stencils for each cell
    for (int binx = nbin_lo[0]; binx <= nbin_hi[0]; ++binx) {
        for (int biny = nbin_lo[1]; biny < nbin_hi[1]; ++biny) {
            for (int binz = nbin_lo[2]; binz < nbin_hi[2]; ++binz) {
                const BinIdx ibin = nbin.hash_bin(binx, biny, binz);

                for (int iz = -1; iz <= 1; ++iz) {
                    for (int iy = -1; iy <= 1; ++iy) {
                        for (int ix = -1; ix <= 1; ++ix) {
                            if ((iz == 1)
                                || ((iz == 0) && (iy == 1))
                                || ((iz == 0) && (iy == 0) && (ix >= 0)))
                            {
                                const int bx = binx + ix;
                                const int by = biny + iy;
                                const int bz = binz + iz;
                                if ((bx >= nbin_lo[0]) && (bx <= nbin_hi[0])
                                    && (by >= nbin_lo[1]) && (by <= nbin_hi[1])
                                    && (bz >= nbin_lo[2]) && (bz <= nbin_hi[2]))
                                {
                                    const BinIdx ibin_next = nbin.hash_bin(bx, by, bz);
                                    stencils_[ibin].emplace_back(ibin_next);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

const std::vector<BinIdx> NStencilHalf::get_stencil(const BinIdx ibin) const {
    return stencils_[ibin];
}

void NStencilHalf::dump(std::ostream& os) const {
    const int mbins = static_cast<int>(stencils_.size());
    for (BinIdx ibin = 0; ibin < mbins; ++ibin) {
        os << ibin << ":";
        for (BinIdx jbin : get_stencil(ibin)) {
            os << " " << jbin;
        }
        os << std::endl;
    }
}

NStencilFull::NStencilFull(const NBin& nbin) {
    const int mbins = nbin.get_mbins();
    stencils_.resize(mbins);

    const auto nbin_size = nbin.get_nbin_size();
    const auto nbin_lo = nbin.get_mbinlo();
    const auto nbin_hi = nbin.get_mbinhi();

    // assign stencils for each cell
    for (int binx = nbin_lo[0]; binx <= nbin_hi[0]; ++binx) {
        for (int biny = nbin_lo[1]; biny < nbin_hi[1]; ++biny) {
            for (int binz = nbin_lo[2]; binz < nbin_hi[2]; ++binz) {
                const BinIdx ibin = nbin.hash_bin(binx, biny, binz);

                // |iz| <= 1 seems to be too narrow..., instead use |iz| <= 2
                for (int iz = -2; iz <= 2; ++iz) {
                    for (int iy = -2; iy <= 2; ++iy) {
                        for (int ix = -2; ix <= 2; ++ix) {
                            const int bx = binx + ix;
                            const int by = biny + iy;
                            const int bz = binz + iz;
                            if ((bx >= nbin_lo[0]) && (bx <= nbin_hi[0])
                                && (by >= nbin_lo[1]) && (by <= nbin_hi[1])
                                && (bz >= nbin_lo[2]) && (bz <= nbin_hi[2]))
                            {
                                const BinIdx ibin_next = nbin.hash_bin(bx, by, bz);
                                stencils_[ibin].emplace_back(ibin_next);
                            }
                        }
                    }
                }
            }
        }
    }
}

const std::vector<BinIdx> NStencilFull::get_stencil(const BinIdx ibin) const {
    return stencils_[ibin];
}
// ----------------------------------------------------------------------------
// getting neighbors
// ----------------------------------------------------------------------------

void get_neighbors(const vector2d& coords, const vector2d& lattice, const double cutoff,
                   vector3d& displacements, vector2i& neighbors)
{
    NBin nbin(lattice, cutoff);
    // NStencilHalf nstencil(nbin);
    NStencilFull nstencil(nbin);

    std::vector<std::vector<double>> coords_with_ghosts(coords);
    const double lx = norm(lattice[0][0], lattice[0][1], lattice[0][2]);
    const double ly = norm(lattice[1][0], lattice[1][1], lattice[1][2]);
    const double lz = norm(lattice[2][0], lattice[2][1], lattice[2][2]);

    // periodic images to fit cutoff radius
    const double EPS = 1e-8;
    const int bxlo = static_cast<int>(floor((-cutoff - EPS * lx) / lx));
    const int bylo = static_cast<int>(floor((-cutoff - EPS * ly) / ly));
    const int bzlo = static_cast<int>(floor((-cutoff - EPS * lz) / lz));
    const int bxhi = static_cast<int>(ceil((lx * (1. + EPS) + cutoff) / lx));
    const int byhi = static_cast<int>(ceil((ly * (1. + EPS) + cutoff) / ly));
    const int bzhi = static_cast<int>(ceil((lz * (1. + EPS) + cutoff) / lz));

    const int nlocal = static_cast<int>(coords.size());
    std::vector<int> origin_atom_mapping;
    for (PeriodicSiteIdx i = 0; i < nlocal; ++i) {
        // for atom in origin cell, PeriodicSiteIdx(i) == i
        origin_atom_mapping.emplace_back(i);
    }

    for (int nx = bxlo; nx <= bxhi; ++nx) {
        for (int ny = bylo; ny <= byhi; ++ny) {
            for (int nz = bzlo; nz <= bzhi; ++nz) {
                if ((nx == 0) && (ny == 0) && (nz == 0)) {
                    // skip origin cell
                    continue;
                }
                for (int i = 0; i < nlocal; ++i) {
                    const double x = coords[i][0] + nx * lattice[0][0] + ny * lattice[1][0] + nz * lattice[2][0];
                    const double y = coords[i][1] + nx * lattice[0][1] + ny * lattice[1][1] + nz * lattice[2][1];
                    const double z = coords[i][2] + nx * lattice[0][2] + ny * lattice[1][2] + nz * lattice[2][2];
                    coords_with_ghosts.emplace_back(std::vector<double>{x, y, z});
                    origin_atom_mapping.emplace_back(i);
                }
            }
        }
    }

    // assign each atom to bin
    nbin.bin_atoms(coords_with_ghosts);

    displacements.resize(nlocal);
    neighbors.resize(nlocal);

    // for atom in origin cell, PeriodicSiteIdx(i) == i
    for (PeriodicSiteIdx i = 0; i < nlocal; ++i) {
        const BinIdx ibin = nbin.get_binidx(i);
        const auto& ri = coords_with_ghosts[i];
        const auto& stencil = nstencil.get_stencil(ibin);

        for (BinIdx bin_neighbor : stencil) {
            const auto& atoms = nbin.get_atoms_within_bin(bin_neighbor);

            for (const PeriodicSiteIdx j : atoms) {
                if ((bin_neighbor == ibin) && (i >= j)) {
                    continue;
                }

                const auto& rj = coords_with_ghosts[j];
                std::vector<double> disp(3);
                disp[0] = rj[0] - ri[0];
                disp[1] = rj[1] - ri[1];
                disp[2] = rj[2] - ri[2];

                const double length = norm(disp[0], disp[1], disp[2]);
                if (length < cutoff) {
                    const int j0 = origin_atom_mapping[j];
                    if (j0 < i) {
                        continue;
                    }

                    displacements[i].emplace_back(disp);
                    neighbors[i].emplace_back(j0);
                }
            }
        }
    }
}


/// this O(N^2) function is left for testing purpose.
void get_neighbors_old(const vector2d& coords, const vector2d& lattice, const double cutoff,
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
