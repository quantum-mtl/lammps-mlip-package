#ifndef MLIPKK_CELL_H_
#define MLIPKK_CELL_H_

#include <iostream>
#include <vector>
#include <memory>

#include "mlipkk_types.h"

namespace MLIP_NS {

/// index for cells with aligned 1-dimensionally
using BinIdx = int;
using PeriodicSiteIdx = int;

class NBin {
    /// # of global bins
    int nbinx_, nbiny_, nbinz_;
    /// global bin sizes
    double binsizex_, binsizey_, binsizez_;

    /// # of local bins (include ghost)
    int mbins_;
    /// bin sizes (include ghost)
    int mbinx_, mbiny_, mbinz_;
    /// lowest global bins
    int mbinxlo_, mbinylo_, mbinzlo_;
    /// highest global bins (not saved in LAMMPS since `mbinxhi == mbinxlo + mbinx - 1`)
    int mbinxhi_, mbinyhi_, mbinzhi_;

    /// bins_[ibin: BinIdx] -> indices of atoms within the `ibin`-th bin
    std::vector<std::vector<PeriodicSiteIdx>> bins_;
    /// atom2bin_[i: PeriodicSiteIdx] returns BinIdx assignment for `i`-th atom
    std::vector<BinIdx> atom2bin_;

    BinIdx OUTSIDE;

public:
    NBin(const std::vector<std::vector<double>>& lattice, const double cutoff);

    void bin_atoms(const std::vector<std::vector<double>>& coords_with_ghosts);

    std::vector<int> get_nbin_size() const;
    std::vector<int> get_mbinlo() const;
    std::vector<int> get_mbinhi() const;
    /// # of bins include cells with ghost atoms
    int get_mbins() const;
    const std::vector<PeriodicSiteIdx>& get_atoms_within_bin(const BinIdx ibin) const;
    BinIdx get_binidx(const PeriodicSiteIdx i) const;

    BinIdx hash_bin(const int binx, const int biny, const int binz) const;

    void dump(std::ostream& os) const;

private:
    /// convert coords to BinIdx
    BinIdx coords2bin(const std::vector<double>& xyz) const;
};

/// For each bin, its stencil is a set of bins which is within cutoff radius
class NStencilHalf {
    std::vector<std::vector<BinIdx>> stencils_;

public:
    NStencilHalf(const NBin& nbin);

    const std::vector<BinIdx> get_stencil(const BinIdx ibin) const;

    void dump(std::ostream& os) const;
};

/// For each bin, its stencil is a set of bins which is within cutoff radius
class NStencilFull {
    std::vector<std::vector<BinIdx>> stencils_;

public:
    NStencilFull(const NBin& nbin);

    const std::vector<BinIdx> get_stencil(const BinIdx ibin) const;
};

/// @param[out] displacements
/// @param[out] neighbors
void get_neighbors_old(const vector2d& coords, const vector2d& lattice, const double cutoff,
                       vector3d& displacements, vector2i& neighbors);

/// @param[out] displacements
/// @param[out] neighbors
void get_neighbors(const vector2d& coords, const vector2d& lattice, const double cutoff,
                   vector3d& displacements, vector2i& neighbors);


inline double norm(const double x, const double y, const double z) {
    return sqrt(x * x + y * y + z * z);
}

} // namespace MLIP_NS

#endif // MLIPKK_CELL_H_
