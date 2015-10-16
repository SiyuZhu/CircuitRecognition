#include <Halide.h>
#include <stdio.h>

using namespace Halide;
using Halide::Image;

#include <fstream>
#include <string>

#include "common.h"
#include "Compiled_Pipeline.h"
#include "../support/image_io.h"
#include "../../utils/read_mnist.h"
#include <boost/date_time/posix_time/posix_time.hpp>

int main(int argc, char** argv) {
  // argv[1] is which image to use
  int num_start = 0;
  int num_stop = 0;
  if (argc > 1)
    num_start = atoi(argv[1]);
  if (argc > 2)
    num_stop = atoi(argv[2]);
  else
    num_stop = num_start;

  printf ("Testing images %d - %d\n", num_start, num_stop);

  // load images
  int num_images, num_labels, rows, cols;
  uint8_t* digits = read_mnist_test_images (num_images, rows, cols);
  uint8_t* labels = read_mnist_test_labels (num_labels);
  assert (num_images == num_labels);

  std::cout << "(" << rows << " x " << cols << ")" << std::endl;

  buffer_t input_buf = {0};

  input_buf.stride[0] = 1;
  input_buf.stride[1] = cols;
  input_buf.stride[2] = rows*cols;

  input_buf.extent[0] = cols;
  input_buf.extent[1] = rows;
  input_buf.extent[2] = 1;
  
  input_buf.elem_size = 1;

  // define the output buffer
  float* output_data = new float[10];
  buffer_t output_buf = {0};
  output_buf.host = (uint8_t*)output_data;

  output_buf.stride[0] = 1;
  output_buf.stride[1] = 10;
  output_buf.stride[2] = 1;

  output_buf.extent[0] = 10;
  output_buf.extent[1] = 1;
  output_buf.extent[2] = 1;

  output_buf.elem_size = sizeof(float);

  input_buf.host = digits;

  // process images
  boost::posix_time::ptime start_cpu_ =
    boost::posix_time::microsec_clock::local_time();
  for(int n = num_start; n <= num_stop; ++n)
    Compiled_Pipeline(&input_buf, &output_buf);
  float elapsed_milliseconds_ = (boost::posix_time::microsec_clock::local_time()
				 - start_cpu_).total_milliseconds();

  // print results
  printf("Total runtime: %g\n", elapsed_milliseconds_);

  delete[] digits;
  delete[] labels;

  return 0;
}

