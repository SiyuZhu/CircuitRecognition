#include "layers.hpp"
#include "Halide.h"
#include <string>
#include <vector>
#include "params.hpp"

using namespace Halide;
using namespace std;

Func convolve(Func input, vector<int> * const& input_dim, Func kernel,
	      const vector<int> * const& kenel_dim, string name) {
  Var x("x"), y("y"), c("c");
  Func result(name);

  RDom r(0, width, 0, height, 0, depth);
  result(x, y, c) = sum(kernel(width-1-r.x, height-1-r.y, r.z, c) *
			input(x+width-1-r.x, y+height-1-r.y, r.z));
  return result;
}

Func convolutional_layer(Func input, Func kernel, Func bias,
			 const int& count, const int& depth, 
			 const int& height, const int& width,
			 string name) {
  Var x("x"), y("y"), c("c");
  Func result = convolve(input, kernel, count, depth, height, width, name);
  result(x, y, c) = result(x, y, c) + bias(c);
  return result;
}

Func maxpool_layer(Func input, const int& height, const int& width, 
		   string name) {
  Var x("x"), y("y"), c("c");
  Func result(name);

  RDom r(0, width, 0, height);
  result(x, y, c) = maximum(input(x * width + r.x, y * height + r.y, c));
  return result;
}

// Inner product
// kernel(n, c, 1, 1)
Func inner_product_layer(Func input, Func weight, Func bias, const int& count,
			 const int& depth, const int& input_height,
			 const int& input_width, string name) {
  Var c("c");
  RDom r(0, input_width, 0, input_height, 0, depth/(input_height*input_width));
  //  RDom r = input.reduction_domain(input.num_update_definitions()-1);
  
  Func result(name);
  if(input.dimensions() == 3) {
    RDom r(0, input_width, 0, input_height, 0, depth/(input_height*input_width));
    result(c) = sum(weight(c, r.x + r.x * r.y + r.x * r.y * r.z, 1, 1) *
		    input(r.x, r.y, r.z)) + bias(c);
  } else {
    RDom r(0, depth);
    result(c) = sum(weight(c, r.x, 1, 1) * input(r.x)) + bias(c);
  }
  return result;
}

// ReLU
Func relu_layer(Func input, string name) {
  Var c("c");

  Func result(name);
  result(c) = max(input(c),0);
  return result;
}

// Softmax
Func softmax_layer(Func input, const int& depth, string name) {
  Var c("c");
  RDom r(0,depth);
  
  Func producer, result(name);  
  producer(c) = exp(input(c));
  result(c) = producer(c) / sum(producer(r.x));
  return result;
}
