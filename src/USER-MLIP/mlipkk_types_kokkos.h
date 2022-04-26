#ifndef MLIPKK_TYPES_KOKKOS_H_
#define MLIPKK_TYPES_KOKKOS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <Kokkos_ScatterView.hpp>
#include <Kokkos_StaticCrsGraph.hpp>

namespace MLIP_NS {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using Layout = ExecSpace::array_layout;
using MemSpace = ExecSpace::memory_space;
using Device = ExecSpace::device_type;

using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using HostLayout = HostExecSpace::array_layout;
using HostDevice = HostExecSpace::device_type;

using range_policy = Kokkos::RangePolicy<ExecSpace>;
using team_policy = Kokkos::TeamPolicy<ExecSpace>;

using view_1d = Kokkos::View<double*, Layout, Device>;
using view_2d = Kokkos::View<double**, Layout, Device>;
using view_3d = Kokkos::View<double***, Layout, Device>;
using view_4d = Kokkos::View<double****, Layout, Device>;
using view_1dc = Kokkos::View<Kokkos::complex<double>*, Layout, Device>;
using view_2dc = Kokkos::View<Kokkos::complex<double>**, Layout, Device>;
using view_3dc = Kokkos::View<Kokkos::complex<double>***, Layout, Device>;
using view_4dc = Kokkos::View<Kokkos::complex<double>****, Layout, Device>;

using StaticCrsGraph = Kokkos::StaticCrsGraph<int, Layout, Device>;
using dview_1i = Kokkos::DualView<int*, Layout, Device>;
using dview_2i = Kokkos::DualView<int**, Layout, Device>;
using dview_1d = Kokkos::DualView<double*, Layout, Device>;
using dview_2d = Kokkos::DualView<double**, Layout, Device>;
using dview_3d = Kokkos::DualView<double***, Layout, Device>;
using dview_4d = Kokkos::DualView<double****, Layout, Device>;
using dview_2dc = Kokkos::DualView<Kokkos::complex<double>**, Layout, Device>;
using dview_3dc = Kokkos::DualView<Kokkos::complex<double>***, Layout, Device>;
using dview_4dc = Kokkos::DualView<Kokkos::complex<double>****, Layout, Device>;
using dview_1p = Kokkos::DualView<Kokkos::pair<int, int>*, Layout, Device>;
using dview_2b = Kokkos::DualView<bool**, Layout, Device>;

using sview_1d = Kokkos::Experimental::ScatterView<double*, Layout, ExecSpace>;
using sview_2d = Kokkos::Experimental::ScatterView<double**, Layout, ExecSpace>;
using sview_3d =
    Kokkos::Experimental::ScatterView<double***, Layout, ExecSpace>;
using sview_4d =
    Kokkos::Experimental::ScatterView<double****, Layout, ExecSpace>;
using sview_4dc = Kokkos::Experimental::ScatterView<Kokkos::complex<double>****,
                                                    Layout, ExecSpace>;

}  // namespace MLIP_NS

#endif  // MLIPKK_TYPES_KOKKOS_H_
