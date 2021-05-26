#include "mlipkk_potential_parser.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

#include "mlipkk_utils.h"
#include "mlipkk_irreps_type.h"
#include "mlipkk_polynomial.h"

namespace MLIP_NS {

void read_potential_file(const char* file, std::vector<std::string>& ele, vector1d& mass,
                         MLIPInput* fp, vector1d& reg_coeffs, Readgtinv& gtinvdata)
{
    std::ifstream input(file);
    if (input.fail()){
        std::cerr << "Error: Could not open mlip file: " << file << std::endl;
        exit(8);
    }

    std::stringstream ss;
    std::string line, tmp;

    // line 1: elements
    ele.clear();
    std::getline(input, line);
    ss << line;
    while (!ss.eof()){
        ss >> tmp;
        ele.push_back(tmp);
    }
    ele.erase(ele.end() - 1);
    ele.erase(ele.end() - 1);
    ss.str("");
    ss.clear(std::stringstream::goodbit);

    fp->n_type = int(ele.size());
    fp->force = true;

    // line 2: cutoff radius
    fp->cutoff = get_value<double>(input);
    // line 3: pair type
    auto pair_type = get_value<std::string>(input);
    if (pair_type == "gaussian") {
        fp->pair_type_id = PairTypeId::GAUSSIAN;
    } else {
        std::cerr << "Unknown pair_type: " << pair_type << std::endl;
        exit(1);
    }
    // line 4: descriptor type
    auto des_type = get_value<std::string>(input);
    if (des_type == "gtinv") {
        fp->des_type_id = DesTypeId::GTINV;
    } else if (des_type == "pair") {
        fp->des_type_id = DesTypeId::PAIR;
        std::cerr << "NotImplemented: des_type == pair" << std::endl;
        exit(1);
    } else {
        std::cerr << "Unknown des_type: " << des_type << std::endl;
        exit(1);
    }

    // line 5-7: model_type, max power, max l
    fp->model_type = get_value<int>(input);
    if (fp->model_type != 2) {
        std::cerr << "NotImplemented: model_type == " << fp->model_type << std::endl;
        exit(1);
    }

    fp->maxp = get_value<int>(input);
    fp->maxl = get_value<int>(input);

    // line 8-10: gtinv_order, gtinv_maxl and gtinv_sym (optional)
    int gtinv_order = 2;
    vector1i gtinv_maxl = {fp->maxl};
    std::vector<bool> gtinv_sym = {false};
    if (fp->des_type_id == DesTypeId::GTINV) {
        gtinv_order = get_value<int>(input);
        const int size = gtinv_order - 1;
        gtinv_maxl = get_value_array<int>(input, size);
        gtinv_sym = get_value_array<bool>(input, size);
    }

    // line 11: number of regression coefficients
    // line 12,13: regression coefficients, scale coefficients
    const int n_reg_coeffs = get_value<int>(input);
    reg_coeffs.clear();
    reg_coeffs = get_value_array<double>(input, n_reg_coeffs);
    const vector1d scale = get_value_array<double>(input, n_reg_coeffs);
    for (int i = 0; i < n_reg_coeffs; ++i) {
        reg_coeffs[i] /= scale[i];
    }

    // line 14: number of gaussian parameters
    // line 15-: gaussian parameters
    int n_params = get_value<int>(input);
    fp->params = vector2d(n_params);
    for (int i = 0; i < n_params; ++i) {
        fp->params[i] = get_value_array<double>(input, 2);
    }

    // last line: atomic mass
    mass.clear();
    mass = get_value_array<double>(input, static_cast<int>(ele.size()));

    gtinvdata = Readgtinv(gtinv_order, gtinv_maxl, gtinv_sym);

}

int get_num_coeffs(const MLIPInput& input, const Readgtinv& gtinvdata) {
    const int n_fn = static_cast<int>(input.params.size());
    const auto& l_array = gtinvdata.get_l_comb();
    const auto irreps_type_pairs = get_unique_irreps_type_pairs(input.n_type, l_array);
    const MLIPPolynomial poly(input.model_type, input.maxp, n_fn, input.n_type, irreps_type_pairs);
    int num_coeffs = static_cast<int>(poly.get_structural_features_prod_index().size());
    return num_coeffs;
}

} // namespace MLIP_NS
