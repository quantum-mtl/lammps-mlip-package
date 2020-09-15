cmake_minimum_required(VERSION 3.10)
project(lammps-mlip-test)

set(CMAKE_CXX_STANDARD 14)
add_compile_options(-Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion)

# boost
find_library(Boost 1.65.1)
include_directories(${Boost_INCLUDE_DIR})

# googletest
add_subdirectory(googletest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(.)

enable_testing()
include_directories(../src/USER-MLIP)
include(GoogleTest)

enable_testing()

add_library(MLIP
    ../src/USER-MLIP/mlip_features.cpp
    ../src/USER-MLIP/mlip_basis_function.cpp
)

add_executable(MLIPFeaturesTest test_mlip_features.cpp)
# add_executable(MLIPFeaturesTest test.cpp)
target_link_libraries(MLIPFeaturesTest gtest gtest_main ${Boost_LIBRALIES} MLIP)
gtest_discover_tests(MLIPFeaturesTest)
