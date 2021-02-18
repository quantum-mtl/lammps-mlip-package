# Required to load the project structure into JETBrain / CLion.
# TODO: Delete at release.

cmake_minimum_required(VERSION 3.17)
project(lammps_mlip_package)

set(CMAKE_CXX_STANDARD 11)
get_filename_component(MLIP_SRC_DIR ${PROJECT_SOURCE_DIR}/src/USER-MLIP ABSOLUTE)
get_filename_component(LAMMPS_SRC_DIR ${PROJECT_SOURCE_DIR}/lammps/src ABSOLUTE)
file(GLOB MLIP_SRC "${MLIP_SRC_DIR}/*.h" "${MLIP_SRC_DIR}/*.cpp")
file(GLOB LAMMPS_SRC "${LAMMPS_SRC_DIR}/*.h" "${LAMMPS_SRC_DIR}/*.cpp")

find_package(Boost 1.65 COMPONENTS program_options REQUIRED)

add_executable(lmp ${LAMMPS_SRC} ${MLIP_SRC})
target_include_directories(lmp PUBLIC ${LAMMPS_SRC_DIR} ${MLIP_SRC_DIR} ${Boost_INCLUDE_DIR})
