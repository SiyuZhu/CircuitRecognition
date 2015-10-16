#include "layers.hpp"
#include "Halide.h"
#include "definitions.hpp"
#include <string>
#include <vector>

using namespace Halide;
using namespace std;

Func maxpool_layer(Func input, vector<Expr> &data_dim, string name) {
  // generate output function
  Var x("x"), y("y"), c("c");
  Func result(name);
  RDom r(0, _MAXPOOL_KERNEL_W_, 0, _MAXPOOL_KERNEL_H_);
  result(x, y, c) = maximum(input(x * _MAXPOOL_KERNEL_W_ + r.x,
				  y * _MAXPOOL_KERNEL_H_ + r.y, c));

  // update output data dimensions
  data_dim[1] /= _MAXPOOL_KERNEL_W_;
  data_dim[0] /= _MAXPOOL_KERNEL_H_;
  
  return result;
}

Func relu_layer(Func input, vector<Expr> &data_dim, string name) {
  // generate output function
  Var x("x");
  Func result(name);
  result(x) = max(input(x),0);
  return result;
}

Func softmax_layer(Func input, vector<Expr> &data_dim, string name) {
  // generate output function
  Var x("x");
  Func producer, result(name);
  RDom r(0,data_dim[0]);
  producer(x) = exp(input(x));
  result(x) = producer(x) / sum(producer(r.x));
  return result;
}
