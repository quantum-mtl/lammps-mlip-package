#ifndef MLIP_POTENTIAL_PARSER_H
#define MLIP_POTENTIAL_PARSER_H

#include <string>
#include <vector>

#include "mlipkk_gtinv_data_reader.h"
#include "mlipkk_types.h"

namespace MLIP_NS {

struct MLIPInput {
    /* number of elements */
    int n_type;
    /* TODO */
    bool force;
    vector2d params;
    /* cutoff radius for neighbor atoms */
    double cutoff;
    /* type of a radial basis function: e.g. "gaussian" */
    char* pair_type;
    /* type of descriptors: "gtinv" or "pair"*/
    char* des_type;
    /* TODO */
    int model_type;
    /* maximum order of polynomial features */
    int maxp;
    /* maximum angular number */
    int maxl;

    ~MLIPInput();
    MLIPInput& operator=(const MLIPInput& other);
};

void read_potential_file(const char* file, std::vector<std::string>& ele, vector1d& mass,
                         MLIPInput* input, vector1d& reg_coeffs, Readgtinv& gtinvdata);

}  // namespace MLIP_NS

#endif
