#ifndef MLIPKK_UTILS_H_
#define MLIPKK_UTILS_H_

#include <complex>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace MLIP_NS {

std::vector<std::vector<int>> get_combinations(int n, int r);
std::vector<std::vector<int>> get_combinations_with_repetition(int n, int r);
std::vector<std::vector<int>> get_combinations_with_repetition_gtinv(int n,
                                                                     int r);

std::vector<double> to_polar_coordinates(const std::vector<double>& xyz);
std::vector<double> to_carterian_coordinates(
    const std::vector<double>& r_polar_azimuthal);

double product_real_part(const std::complex<double>& lhs,
                         const std::complex<double>& rhs);

template <typename T>
T get_value(std::ifstream& input) {
    std::string line;
    std::stringstream ss;

    T val;
    std::getline(input, line);
    ss << line;
    ss >> val;

    return val;
}

template <typename T>
std::vector<T> get_value_array(std::ifstream& input, const int& size) {
    std::string line;
    std::stringstream ss;

    std::vector<T> array(size);

    std::getline(input, line);
    ss << line;
    T val;
    for (int i = 0; i < static_cast<int>(array.size()); ++i) {
        ss >> val;
        array[i] = val;
    }

    return array;
}

}  // namespace MLIP_NS

#endif  // MLIPKK_UTILS_H_
