#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include <Halide.h>

using namespace Halide;

template <typename T>
Func convolve (
    Func img, 
    Func kernel, 
    const int kx, const int ky, const int num_img, const int num_kern, 
    std::string name
  );

#include "convolution.cpp"

#endif
