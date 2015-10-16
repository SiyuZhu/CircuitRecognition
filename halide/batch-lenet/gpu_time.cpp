#include <stdio.h>
#include "mnist_gpu_pipeline.h"
#include "../../utils/read_mnist.h"
#include "buffer_io.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <vector>
#include <iostream>

using namespace std;

// uses the mnist pipeline to classify the given images
// the first argument specifies the batch size, and the second argument
// specifies the number of iterations
int main(int argc, char** argv) {
  // batch size and iterations
  int batch_size = atoi(argv[1]);
  int iterations = atoi(argv[2]);

  // load test data
  int count, rows, cols;
  uint8_t *test_labels = read_mnist_test_labels(count);
  uint8_t *test_images = read_mnist_test_images(count, rows, cols);
  buffer_t test_images_buf{0};
  init_buffer(test_images_buf, cols, rows, 1, count, 1, test_images);

  // load output buffer
  buffer_t output_buf{0};
  init_buffer(output_buf, 10, batch_size, 0, 0, sizeof(float),(uint8_t *) (new float[10*batch_size]));

  // computing time is the amount of time spent on the pipeline. 
  // it should be noted that caffe benchmarks are based on computing time
  float compute_time = 0;
  float total_time = 0;
  boost::posix_time::ptime compute_start;
  boost::posix_time::ptime compute_end;
  boost::posix_time::ptime total_end;
  boost::posix_time::ptime total_start = 
    boost::posix_time::microsec_clock::local_time();

  // classify test images
  for(int n = 0; n < iterations; ++n) {
    // select the image to classify
    buffer_t masked_image_buf = test_images_buf;
    dimension_mask(masked_image_buf, 3, n*batch_size, batch_size);
    masked_image_buf.host_dirty = true;

    // run pipeline
    compute_start = boost::posix_time::microsec_clock::local_time();
    mnist_gpu_pipeline(&masked_image_buf, &output_buf);
    compute_end = boost::posix_time::microsec_clock::local_time();
    compute_time += (compute_end - compute_start).total_milliseconds();
  }

  // print performance results
  total_end = boost::posix_time::microsec_clock::local_time();
  total_time = (total_end - total_start).total_milliseconds();
  printf("Total time: %g\n", total_time);
  printf("Computing time: %g\n", compute_time);

  delete[] output_buf.host;
  delete[] test_images;
  delete[] test_labels;

  return 0;
}


