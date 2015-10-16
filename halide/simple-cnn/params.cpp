#ifndef PARAMS_CPP
#define PARAMS_CPP

#include "params.h"
#include "../../utils/read_params.h"

void Parameters::read_from_file (
    std::string file_name, 
    int dense_flag
) {
  read_params(
      file_name, 
      weight_data, bias_data, 
      layers, count, width, height, 
      dense_flag
  );
}


void Parameters::print_to_file(std::string file_name){
    std::string weight_name = file_name + "_weight";
    std::string bias_name = file_name + "_bias";

    print_func<float>(weight(weight_name), width, height, layers, count, weight_name);
    print_func<float>(bias(bias_name), 1, 1, 1, count, bias_name);
}

#endif
