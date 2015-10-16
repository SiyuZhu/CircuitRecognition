#include "layers.hpp"
#include "Halide.h"
#include "definitions.hpp"
#include <string>
#include <vector>

using namespace Halide;
using namespace std;

Func maxpool_layer(Func input, vector<Expr> &data_dim, string name) {
  // generate output function
  Var x("x"), y("y"), c("c"), n("n");
  Func result(name);
  RDom r(0, _MAXPOOL_KERNEL_W_, 0, _MAXPOOL_KERNEL_H_);
  result(x, y, c, n) = maximum(input(x * _MAXPOOL_KERNEL_W_ + r.x,
				  y * _MAXPOOL_KERNEL_H_ + r.y, c, n));

  // update output data dimensions
  data_dim[1] /= _MAXPOOL_KERNEL_W_;
  data_dim[0] /= _MAXPOOL_KERNEL_H_;
  
  return result;
}

Func relu_layer(Func input, vector<Expr> &data_dim, string name) {
  // generate output function
  Var x("x"), n("n");
  Func result(name);
  result(x, n) = max(input(x, n),0);
  return result;
}

Func softmax_layer(Func input, vector<Expr> &data_dim, string name) {
  // generate output function
  Var x("x"), n("n");
  Func producer, result(name);
  RDom r(0,data_dim[0]);
  producer(x, n) = exp(input(x,n));
  result(x, n) = producer(x, n) / sum(producer(r.x, n));
  return result;
}
