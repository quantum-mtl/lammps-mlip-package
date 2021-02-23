#ifndef LMP_MLIP_MODEL_H
#define LMP_MLIP_MODEL_H

#include <vector>
#include <string>
#include <fstream>

#include "mlip_pymlcpp.h"
#include "mlip_model_params.h"
#include "mlip_polynomial_gtinv.h"

namespace MLIP_NS {

class DataMLIP {
    feature_params fp_;
    ModelParams modelp_;
    vector2i lm_info_;
    PolynomialGtinv poly_gtinv_;
    vector1d reg_coeffs_;
    std::vector<std::string> ele_;
    vector1d mass_;
    vector2i type_comb_;

public:
    DataMLIP() = default;
    ~DataMLIP() = default;

    void initialize(char*);

    const feature_params& get_feature_params() const;
    const ModelParams& get_model_params() const;
    const vector2i& get_lm_info() const;
    const PolynomialGtinv& get_poly_gtinv() const;
    const vector1d& get_reg_coeffs() const;
    const std::vector<std::string>& get_elements() const;
    const vector1d& get_masses() const;
    const vector2i& get_type_comb() const;

private:
    template<typename T> T get_value(std::ifstream&);
    template<typename T> std::vector<T> get_value_array(std::ifstream&, const int&);
};

} // namespace MLIP_NS

#endif // LMP_MLIP_MODEL_H
