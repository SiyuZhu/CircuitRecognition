#ifndef PARAMS_H
#define PARAMS_H

#include <Halide.h>
using namespace Halide;

#include "common.h"

// To record parameters of each layer including the size of parameter matrixs
//template<typename T>
class Parameters{
  float *weight_data, *bias_data;
  public:
    // these data members are public for now
    int width, height, layers, count;

    // constructor
    Parameters() : 
      weight_data(NULL),
      bias_data(NULL),
      width(0), height(0),
      layers(0), count(0)
    {
    }

    // copy constructor
    Parameters(const Parameters& other) :
      width(other.width), height(other.height),
      layers(other.layers), count(other.count)
    {
      if (weight_data != NULL) {
        int weight_size = width*height*layers*count;
        weight_data = new float[weight_size];
        memcpy (weight_data, other.weight_data, weight_size*sizeof(float));
      }
      if (bias_data != NULL) {
        int bias_size = count;
        bias_data = new float[bias_size];
        memcpy (bias_data, other.bias_data, bias_size*sizeof(float));
      }
    }

    // destructor
    ~Parameters(){
      delete[] weight_data;
      delete[] bias_data;
    } 

    //Read parameter function
    void read_from_file(std::string file_name, int dense_flag);

    //Print the parameter data file out for debugging
    void print_to_file(std::string file_name);

    //Generate the weight Func, not sure whether this string_name argument is appropriate
    Func weight(std::string string_name){
      return Halide_func_from_buffer(width, height, layers, count, weight_data, string_name);
    }
    //Generate the bias Func
    Func bias(std::string string_name){
      return Halide_func_from_buffer(1, 1, 1, count, bias_data, string_name);
    }
};

#endif

