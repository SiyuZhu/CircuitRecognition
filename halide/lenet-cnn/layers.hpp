#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "Halide.h"
#include "definitions.hpp"
#include <string>
#include <vector>

using namespace Halide;
using namespace std;

// IMPORTANT:
// data_dim stores dimension sizes using the same organization as halide.
// data_dim are required by each function because the data output of the 
// maxpool layer has a shape that is incompatible with the shape of the inner
// product weights, so the data dimensions must be tracked to make a proper
// reduction domain for the inner_product layer. it would also be useful for
// debugging purposes.

// Requires 3 dimensional input, 4 dimensional weight, 1 dimensional bias
// Returns 3 dimensional output
template <typename T>
Func convolutional_layer(Func input, vector<Expr> &data_dim, Image<T> &weight,
			 Image<T> &bias, string name){
  // generate output function
  Var x("x"), y("y"), c("c");
  Func result(name);
  RDom r(0, weight.extent(0), 0, weight.extent(1), 0, weight.extent(2));
  result(x, y, c) = bias(c) + sum(weight(weight.extent(0)-1-r.x,
					 weight.extent(1)-1-r.y, r.z, c) *
				  input(x+weight.extent(0)-1-r.x,
					y+weight.extent(1)-1-r.y, r.z));

  // update output data dimensions
  data_dim[2] = weight.extent(3);
  data_dim[1] = (data_dim[1] + 2*_PAD_H_ - weight.extent(1))/_STRIDE_H_ + 1;
  data_dim[0] = (data_dim[0] + 2*_PAD_W_ - weight.extent(0))/_STRIDE_W_ + 1;
  return result;
}

// Requires 3 dimensional input
// Returns 3 dimensional output
Func maxpool_layer(Func input, vector<Expr> &data_dim, string name);

// Requires 1 dimensional or 3 dimensional input
// Returns 1 dimensional output
template <typename T>
Func innerproduct_layer(Func input, vector<Expr> &data_dim, Image<T> &weight,
			Image<T> &bias, string name) {
  // generate output function
  Var x("x");
  Func result(name);
  if(data_dim.size() == 3) {
    RDom r(0, data_dim[0], 0, data_dim[1], 0, data_dim[2]);
    result(x) = sum(weight(x, r.x + data_dim[0]*r.y + 
   			   data_dim[0]*data_dim[1]*r.z, 0, 0) *
      		    input(r.x, r.y, r.z)) + bias(x);
  } else {
    RDom r(0, data_dim[0]);
    result(x) = sum(weight(x, r.x, 0, 0) * input(r.x)) + bias(x);
  }

  // update ouput data dimensions
  data_dim.resize(1);
  data_dim[0] = bias.extent(0);

  return result;
}

// Requires 1 dimensional input
// Returns 1 dimensional output
Func relu_layer(Func input, vector<Expr> &data_dim, string name);

// Requires 1 dimensional input
// Returns 1 dimensional output
Func softmax_layer(Func input, vector<Expr> &data_dim, string name);

#endif
