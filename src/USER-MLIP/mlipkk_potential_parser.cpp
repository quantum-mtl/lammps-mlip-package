#include "mlipkk_potential_parser.h"

#include <iostream>
#include <fstream>
#include <cassert>
#include <cstring>

#include "mlipkk_utils.h"

namespace MLIP_NS {

MLIPInput::~MLIPInput() {
    delete [] pair_type;
    delete [] des_type;
}

MLIPInput& MLIPInput::operator=(const MLIPInput& other) {
    if (this != &other) {
        n_type = other.n_type;
        force = other.force;
        params = other.params;
        cutoff = other.cutoff;
        model_type = other.model_type;
        maxp = other.maxp;
        maxl = other.maxl;

        pair_type = new char [strlen(other.pair_type) + 1];
        strcpy(pair_type, other.pair_type);

        des_type = new char [strlen(other.des_type) + 1];
        strcpy(des_type, other.des_type);
    }

    return *this;
}

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

    // line 2-4: cutoff radius, pair type, descriptor type
    fp->cutoff = get_value<double>(input);
    auto pair_type = get_value<std::string>(input);
    fp->pair_type = new char [pair_type.size() + 1];
    strcpy(fp->pair_type, pair_type.c_str());
    auto des_type = get_value<std::string>(input);
    fp->des_type = new char [des_type.size() + 1];
    strcpy(fp->des_type, des_type.c_str());

    // line 5-7: model_type, max power, max l
    fp->model_type = get_value<int>(input);
    fp->maxp = get_value<int>(input);
    fp->maxl = get_value<int>(input);

    // line 8-10: gtinv_order, gtinv_maxl and gtinv_sym (optional)
    char accepted_des_type[] = "gtinv";
    // TODO: add des_type=="pair"
    assert(strcmp(fp->des_type, accepted_des_type) == 0);
    int gtinv_order = get_value<int>(input);
    int size = gtinv_order - 1;
    vector1i gtinv_maxl = get_value_array<int>(input, size);
    std::vector<bool> gtinv_sym = get_value_array<bool>(input, size);

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

} // namespace MLIP_NS
