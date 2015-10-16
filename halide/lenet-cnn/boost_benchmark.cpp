#include <stdio.h>
#include "mnist_pipeline.h"
#include <boost/date_time/posix_time/posix_time.hpp> 
#include "../../utils/read_mnist.h"
#include "buffer_io.hpp"

using namespace std;

// uses the mnist pipeline to classify the given images
// the first argument specifies the first image to classify, and the second
// argument specifies the last image to classify
int main(int argc, char** argv) {
  // parse image index arguments
  int num_start = 0;
  int num_stop = 0;
  if (argc > 1)
    num_start = atoi(argv[1]);
  if (argc > 2)
    num_stop = atoi(argv[2]);
  else
    num_stop = num_start;
  printf ("Testing images %d - %d\n", num_start, num_stop);

  // load test data
  int count, rows, cols;
  uint8_t *test_labels = read_mnist_test_labels(count);
  uint8_t *test_images = read_mnist_test_images(count, rows, cols);
  buffer_t test_images_buf{0};
  init_buffer(test_images_buf, cols, rows, 1, 1, 1, test_images);

  // load output buffer
  buffer_t output_buf{0};
  init_buffer(output_buf, 10, 0, 0, 0, sizeof(float),(uint8_t *) (new float[10]));

  // process images
  boost::posix_time::ptime start_cpu_ =
    boost::posix_time::microsec_clock::local_time();
  for(int n = num_start; n <= num_stop; ++n)
    mnist_pipeline(&test_images_buf, &output_buf);
  float elapsed_milliseconds_ = (boost::posix_time::microsec_clock::local_time()
				 - start_cpu_).total_milliseconds();

  // print results
  printf("Total runtime: %g\n", elapsed_milliseconds_);

  delete[] output_buf.host;
  delete[] test_images;
  delete[] test_labels;

  return 0;
}

