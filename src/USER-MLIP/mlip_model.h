#ifndef LMP_MLIP_MODEL_H
#define LMP_MLIP_MODEL_H

#include <vector>
#include <string>
#include <fstream>

#include "mlip_pymlcpp.h"
#include "mlip_model_params.h"
#include "mlip_polynomial_gtinv.h"
#include "mlip_polynomial_pair.h"
#include "mlip_read_gtinv.h"
#include "mlip_features.h"

namespace MLIP_NS {

/// @brief Class for managing hyperparameter and regression coefficients of potential model
/// @tparam POLY polynomial structural feature
template<typename POLY>
class DataMLIPBase {
    feature_params fp_;
    ModelParams modelp_;
    vector2i lm_info_;
    POLY poly_feature_;
    vector1d reg_coeffs_;
    std::vector<std::string> ele_;
    vector1d mass_;
    vector2i type_comb_;

public:
    DataMLIPBase() = default;
    ~DataMLIPBase() = default;

    void initialize(char* file) {
        std::ifstream input(file);
        if (input.fail()){
            std::cerr << "Error: Could not open mlip file: " << file << "\n";
            exit(8);
        }

        std::stringstream ss;
        std::string line, tmp;

        // line 1: elements
        ele_.clear();
        std::getline(input, line);
        ss << line;
        while (!ss.eof()){
            ss >> tmp;
            ele_.push_back(tmp);
        }
        ele_.erase(ele_.end() - 1);
        ele_.erase(ele_.end() - 1);
        ss.str("");
        ss.clear(std::stringstream::goodbit);

        fp_.n_type = int(ele_.size());
        fp_.force = true;

        // line 2-4: cutoff radius, pair type, descriptor type
        // line 5-7: model_type, max power, max l
        fp_.cutoff = get_value<double>(input);
        fp_.pair_type = get_value<std::string>(input);
        fp_.des_type = get_value<std::string>(input);
        fp_.model_type = get_value<int>(input);
        fp_.maxp = get_value<int>(input);
        fp_.maxl = get_value<int>(input);

        // line 8-10: gtinv_order, gtinv_maxl and gtinv_sym (optional)
        if (fp_.des_type == "gtinv"){
            int gtinv_order = get_value<int>(input);
            int size = gtinv_order - 1;
            vector1i gtinv_maxl = get_value_array<int>(input, size);
            std::vector<bool> gtinv_sym = get_value_array<bool>(input, size);

            Readgtinv rgt(gtinv_order, gtinv_maxl, gtinv_sym);
            fp_.lm_array = rgt.get_lm_seq();
            fp_.l_comb = rgt.get_l_comb();
            fp_.lm_coeffs = rgt.get_lm_coeffs();
            lm_info_ = get_lm_info_table(fp_.maxl);
        }

        // line 11: number of regression coefficients
        // line 12,13: regression coefficients, scale coefficients
        int n_reg_coeffs = get_value<int>(input);
        reg_coeffs_ = get_value_array<double>(input, n_reg_coeffs); // TODO: move semantics
        vector1d scale = get_value_array<double>(input, n_reg_coeffs);
        for (int i = 0; i < n_reg_coeffs; ++i) reg_coeffs_[i] *= 2.0/scale[i];

        // line 14: number of gaussian parameters
        // line 15-: gaussian parameters
        int n_params = get_value<int>(input);
        fp_.params = vector2d(n_params);
        for (int i = 0; i < n_params; ++i) {
            fp_.params[i] = get_value_array<double>(input, 2);
        }

        // last line: atomic mass
        mass_ = get_value_array<double>(input, ele_.size());

        modelp_ = ModelParams(fp_);
        // TODO: when POLY=PolynomialPair, poly_feature_ is created if fp_.maxp > 1 in origianl code.
        // Was the if-clause needed?
        poly_feature_ = POLY(fp_, modelp_, lm_info_);

        type_comb_ = vector2i(fp_.n_type, vector1i(fp_.n_type));
        for (int type1 = 0; type1 < fp_.n_type; ++type1){
            for (int type2 = 0; type2 < fp_.n_type; ++type2){
                for (int i = 0; i < modelp_.get_type_comb_pair().size(); ++i){
                    const auto &tc = modelp_.get_type_comb_pair()[i];
                    if (tc[type1].size() > 0 and tc[type1][0] == type2){
                        type_comb_[type1][type2] = i;
                        break;
                    }
                }
            }
        }
    }

    const feature_params& get_feature_params() const {
        return fp_;
    }

    const ModelParams& get_model_params() const {
        return modelp_;
    }

    const vector2i& get_lm_info() const {
        return lm_info_;
    }

    const POLY& get_poly_feature() const {
        return poly_feature_;
    }

    const vector1d& get_reg_coeffs() const {
        return reg_coeffs_;
    }

    const std::vector<std::string>& get_elements() const {
        return ele_;
    }

    const vector1d& get_masses() const {
        return mass_;
    }

    const vector2i& get_type_comb() const {
        return type_comb_;
    }

    double get_cutmax() const {
        return get_feature_params().cutoff;
    }

private:
    template<typename T>
    T get_value(std::ifstream& input ) {
        std::string line;
        std::stringstream ss;

        T val;
        std::getline( input, line );
        ss << line;
        ss >> val;

        return val;
    }

    template<typename T>
    std::vector<T> get_value_array
    (std::ifstream& input, const int& size) {
        std::string line;
        std::stringstream ss;

        std::vector<T> array(size);

        std::getline( input, line );
        ss << line;
        T val;
        for (int i = 0; i < array.size(); ++i){
            ss >> val;
            array[i] = val;
        }

        return array;
    }
};

} // namespace MLIP_NS

#endif // LMP_MLIP_MODEL_H
