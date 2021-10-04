if(CMAKE_CXX_STANDARD LESS 17)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Enable lambda
if(Kokkos_ENABLE_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --expt-extended-lambda")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
# set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MLIP_INCLUDE_DIR ${LAMMPS_SOURCE_DIR}/USER-MLIP)
file(GLOB MLIP_SOURCES ${LAMMPS_SOURCE_DIR}/USER-MLIP/mlip*.cpp)

# Kokkos headers
if(NOT DEFINED Kokkos_INCLUDE_DIRS)
  set(LAMMPS_LIB_KOKKOS_SRC_DIR ${LAMMPS_LIB_SOURCE_DIR}/kokkos)
  set(LAMMPS_LIB_KOKKOS_BIN_DIR ${LAMMPS_LIB_BINARY_DIR}/kokkos)
  set(Kokkos_INCLUDE_DIRS ${LAMMPS_LIB_KOKKOS_SRC_DIR}/core/src
                          ${LAMMPS_LIB_KOKKOS_SRC_DIR}/containers/src
                          ${LAMMPS_LIB_KOKKOS_SRC_DIR}/algorithms/src
                          ${LAMMPS_LIB_KOKKOS_BIN_DIR})
endif()

add_library(mlip_gtinv STATIC ${MLIP_SOURCES})
set_target_properties(mlip_gtinv PROPERTIES CXX_EXTENSIONS ON OUTPUT_NAME lammps_mlip${LAMMPS_MACHINE})
target_include_directories(mlip_gtinv PUBLIC ${MLIP_INCLUDE_DIR} ${Kokkos_INCLUDE_DIRS})
target_link_libraries(lammps PRIVATE mlip_gtinv)
