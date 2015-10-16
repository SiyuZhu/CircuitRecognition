#include <Halide.h>
#include <stdio.h>

using namespace Halide;
using Halide::Image;

#include "../support/image_io.h"
#include "../../utils/timer.h"

// returns a function which is the conv result
template <typename T>
Func convolve3D (
    Func img, 
    Func kernel, 
    const int kw, const int kh, const int depth, 
    const char* name
  )
{
  Var x("x"), y("y"), c("c"), xi("xi"), yo("yo"), yi("yi");
  Func result(name);

#if 0
  // pure def
  result(x,y,c) = cast<T>(0);
  // update 0
  RDom r(0,kw, 0,kh, 0,depth);
  result(x,y,c) += kernel(kw-1-r.x, kh-1-r.y, r.z) * img(x+kw-1-r.x, y+kh-1-r.y, r.z);
  //result.update(0).tile(x,y,xi,yi,3,3);
#elif 1
  //result(x,y,c) = cast<T>(0);
  RDom r(0,kw, 0,kh);
  result(x,y,c) = sum(kernel(kw-1-r.x, kh-1-r.y, c) * img(x+kw-1-r.x, y+kh-1-r.y, c));
  //result.tile(x,y,xi,yi,100,50);
#endif

  return result;
}

int main(int argc, char** argv) {
  Var i("i"), j("j"), c("c");

  const int fcols= 3, frows = 3, cols = 21, rows = 21;

  // Create the kernel
  uint8_t kernel_data[fcols*frows] = {
    2,0,0,
    0,0,0,
    0,0,0
  };

  // wrap a Buffer around the array, then wrap an Image around that
  Image<uint8_t> kernel_img ( Buffer(UInt(8), fcols,frows,1,0, kernel_data, "kernel_img") );
  print_image(kernel_img, "kernel.dat");

  // Create the Input image
  uint8_t input_data[rows*cols];
  for (int i = 0; i < rows; ++i) 
    for (int j = 0; j < cols; ++j)
      input_data[i*cols+j] = i+1;
  Image<uint8_t> input_img ( Buffer(UInt(8), cols,rows,1,0, input_data, "input_img") );
  print_image(input_img, "input.dat");

  Func input("input");
  input(i,j,c) = input_img(i,j,c);
  
  Func kernel("kernel");
  kernel(i,j,c) = kernel_img(i,j,c);

  Func output = convolve3D<uint8_t>(input, kernel, fcols, frows, 1, "output");

  Image<uint8_t> output_img(cols-fcols+1, rows-frows+1, input_img.channels(), "output_img");

  Timer t1("output");
  t1.start();
  output.realize(output_img);
  t1.stop();
  print_image(output_img, "output.dat");

  printf("Success!\n");
  return 0;
}
