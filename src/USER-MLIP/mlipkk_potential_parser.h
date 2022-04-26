#ifndef MLIPKK_POTENTIAL_PARSER_H_
#define MLIPKK_POTENTIAL_PARSER_H_

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
    PairTypeId pair_type_id;
    /* type of descriptors: "gtinv" or "pair"*/
    DesTypeId des_type_id;
    /* TODO */
    int model_type;
    /* maximum order of polynomial features */
    int maxp;
    /* maximum angular number */
    int maxl;

    ~MLIPInput() = default;
};

void read_potential_file(const char* file, std::vector<std::string>& ele,
                         vector1d& mass, MLIPInput* input, vector1d& reg_coeffs,
                         Readgtinv& gtinvdata);

int get_num_coeffs(const MLIPInput& input, const Readgtinv& gtinvdata);

}  // namespace MLIP_NS

#endif  // MLIPKK_POTENTIAL_PARSER_H_
