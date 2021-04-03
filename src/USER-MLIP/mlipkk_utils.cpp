#include "mlipkk_utils.h"

#include <vector>
#include <queue>
#include <cmath>

namespace MLIP_NS {

/// @brief chose k elements from {0, 1, ..., n - 1} without repetitions
std::vector<std::vector<int>> get_combinations(int n, int r) {
    std::vector<std::vector<int>> ret;

    std::queue<std::vector<int>> que;
    que.push(std::vector<int>());
    while (!que.empty()) {
        auto comb = que.front(); que.pop();

        if (static_cast<int>(comb.size()) == r) {
            ret.emplace_back(comb);
            continue;
        }

        int last = -1;
        if (!comb.empty()) {
            last = comb[comb.size() - 1];
        }
        for (int i = last + 1; i < n; ++i) {
            std::vector<int> next_comb(comb);
            next_comb.emplace_back(i);
            que.push(next_comb);
        }
    }

    return ret;
}

/// @brief chose k elements from {0, 1, ..., n - 1} with repetitions
std::vector<std::vector<int>> get_combinations_with_repetition(int n, int r) {
    std::vector<std::vector<int>> ret;

    std::queue<std::vector<int>> que;
    que.push(std::vector<int>());
    while (!que.empty()) {
        auto comb = que.front(); que.pop();

        if (static_cast<int>(comb.size()) == r) {
            ret.emplace_back(comb);
            continue;
        }

        int last = 0;
        if (!comb.empty()) {
            last = comb[comb.size() - 1];
        }
        for (int i = last; i < n; ++i) {
            std::vector<int> next_comb(comb);
            next_comb.emplace_back(i);
            que.push(next_comb);
        }
    }

    return ret;
}

/// @brief chose k elements from {0, 1, ..., n - 1} with repetitions.
///        Used for constructing indices of polynomial features in historial reason.
std::vector<std::vector<int>> get_combinations_with_repetition_gtinv(int n, int r) {
    std::vector<std::vector<int>> ret;

    std::queue<std::vector<int>> que;
    for (int first = 0; first < n; ++first) {
        que.push(std::vector<int>{first});
        while (!que.empty()) {
            auto comb = que.front(); que.pop();

            if (static_cast<int>(comb.size()) == r) {
                ret.emplace_back(comb);
                continue;
            }

            int last = comb[comb.size() - 1];
            for (int i = 0; i <= last; ++i) {
                std::vector<int> next_comb(comb);
                next_comb.emplace_back(i);
                que.push(next_comb);
            }
        }
    }

    return ret;
}

std::vector<double> to_polar_coordinates(const std::vector<double>& xyz) {
    const double x = xyz[0];
    const double y = xyz[1];
    const double z = xyz[2];

    const double r = sqrt(x * x + y * y + z * z);
    const double polar = acos(z / r);
    const double azimuthal = atan2(y, x);

    std::vector<double> r_polar_azimuthal {r, polar, azimuthal};
    return r_polar_azimuthal;
}

std::vector<double> to_carterian_coordinates(const std::vector<double>& r_polar_azimuthal) {
    const double r = r_polar_azimuthal[0];
    const double polar = r_polar_azimuthal[1];
    const double azimuthal = r_polar_azimuthal[2];

    const double x = r * sin(polar) * cos(azimuthal);
    const double y = r * sin(polar) * sin(azimuthal);
    const double z = r * cos(polar);

    std::vector<double> xyz {x, y, z};
    return xyz;
}

double product_real_part(const std::complex<double>& lhs, const std::complex<double>& rhs) {
    return lhs.real() * rhs.real() - lhs.imag() * rhs.imag();
}

} // namespace MLIP_NS
