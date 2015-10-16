#ifndef PARAMS_HPP
#define PARAMS_HPP

#include "Halide.h"
#include <string>
#include <vector>
#include <numeric>
#include <functional>

using namespace Halide;
using namespace std;

// To record parameters of each layer including the size of parameter matrixs
template<typename T>
class Parameters{
  int bias_size_;
  // weight_dim_ contains the dimensions of the c-style array storing weights.
  // When iterating through the array, the index in the last dimension changes 
  // the fastest.
  vector<int> *weight_dim_;
  T *weight_data, *bias_data;

public:
  Parameters(): bias_size_(0), weight_dim_(NULL), weight_data(NULL), bias_data(NULL) {}
  Parameters(const Parameters<T>& other): bias_size_(other.bias_size_),
					  weight_dim_(new vector<int>(*other.weight_dim_)) {
    if(other.weight_data != NULL) {
      int weight_size = accumulate(weight_dim_->begin(), weight_dim_->end(),
				   1, multiplies<int>());
      weight_data = new T[weight_size];
      memcpy(weight_data, other.weight_data, weight_size*sizeof(T));
    }
    if(other.bias_data != NULL) {
      bias_data = new T[bias_size_];
      memcpy (bias_data, other.bias_data, bias_size*sizeof(T));
    }
  }
  
  ~Parameters() {
    delete weight_dim_;
    delete[] weight_data;
    delete[] bias_data;
  }

  // Var a refers to the index in the last dimension of weight_dim_
  Func weight_func(string name, Var a, Var b, Var c, Var d) {
    // in buffer arguments, the index of the first dimension whose size is 
    // proveded changes fastest in a c-style array iteration (thus it
    // corresponds to the value of the 3rd weight dimension
    Image<T> image = Image<T>(Buffer(type_of<T>(), (*weight_dim_)[3],
				     (*weight_dim_)[2], (*weight_dim_)[1],
				     (*weight_dim_)[0]));
    Func F(name);
    F(a, b, c, d) = cast<T>(image(a, b, c, d));
    return F;
  }

  Func bias_func(string name, Var a) {
    Image<T> image = Image<T>(Buffer(type_of<T>(), bias_size_, 0, 0, 0,
				     (uint8_t*) bias_data));
    Func F(name);
    F(a) = cast<T>(image(a));
    return F;
  }

  const int& bias_size() {return bias_size_;}

  const vector * const& weight_dim() {return weight_dim_;}

  // .dat file stores conv params in n*c*h*w c-style array and
  // ip params in 1*1*c*n c-stype array.
  static Parameters<float> *dat_Factory(string filename, int dense_flag) {
    Parameters<float> *params = new Parameters<float>;
    params->weight_dim_ = new vector<int>(4);
    if(dense_flag == 1) {
      read_params(filename, params->weight_data, params->bias_data,
		  (*params->weight_dim_)[2], (*params->weight_dim_)[3],
		  (*params->weight_dim_)[1], (*params->weight_dim_)[0],
		  dense_flag);
      params->bias_size_ = (*params->weight_dim_)[3];
    } else {
      read_params(filename, params->weight_data, params->bias_data,
		  (*params->weight_dim_)[1], (*params->weight_dim_)[0],
		  (*params->weight_dim_)[2], (*params->weight_dim_)[3],
		  dense_flag);
      params->bias_size_ = (*params->weight_dim_)[0];
    }
    return params;
  }
};

#endif
