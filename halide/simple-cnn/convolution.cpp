#ifndef CONVOLUTION_CPP
#define CONVOLUTION_CPP

#include "convolution.h"
#include "common.h"

using namespace Halide;

// Convolves an array of images (Width, Height, num_images) with
// an array of kernels (kWidth, kHeight, num_images, num_kernels) to generate
// an array of results (Width-kWidth+1, Height-kHeight+1, num_kernels)
//        
// returns a function which is the conv result
template <typename T>
Func convolve (
    Func img, 
    Func kernel, 
    const int kx, const int ky, const int num_img, const int num_kern,
    std::string name
  )
{
  Var x("x"), y("y"), c("c");
  Func result(name);

  /* Old method
  // pure def
  result(x,y,c) = cast<T>(0);
  // update 0
  RDom r(0,kx, 0,ky, 0,kz);
  result(x,y,c) += kernel(kx-1-r.x, ky-1-r.y, r.z) * img(x+kx-1-r.x, y+ky-1-r.y, r.z);
  //result.update(0).tile(x,y,xi,yi,3,3);
  */

  RDom r(0,kx, 0,ky, 0,num_img);
  result(x,y,c) = sum(kernel(kx-1-r.x, ky-1-r.y, r.z, c) * img(x+kx-1-r.x, y+ky-1-r.y, r.z));

  return result;
}

// Performs max pooling
template <typename T>
Func maxpool (
    Func img,
    const int kx, const int ky, const int num_img,
    std::string name
  )
{
  Var x("x"), y("y"), c("c");
  Func result(name);
  
  RDom m(0, kx, 0, ky, 0, num_img);
  result(x,y,c) = Halide::maximum(img(x*kx+m.x, y*ky+m.y, c));

  return result;
}

#endif
