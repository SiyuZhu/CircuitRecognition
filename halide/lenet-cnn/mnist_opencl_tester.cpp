#include <stdio.h>
#include "mnist_opencl_pipeline.h"
#include "../../utils/timer.h"
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
  init_buffer(test_images_buf, cols, rows, 1, count, 1, test_images);

  // load output buffer
  buffer_t output_buf{0};
  init_buffer(output_buf, 10, 0, 0, 0, sizeof(float),(uint8_t *)( new float[10]));
  
  // TODO by setting host dirty on every iteration, is data being copied
  // unnecessarily (since the actual input data does not change, but only
  // the mask changes).
  // classify test images
  int correct = 0;
  Timer tm("lenet-cnn");
  for (int n = num_start; n <= num_stop; ++n) {
    // select the image to classify
    buffer_t masked_image_buf{0};
    init_buffer(masked_image_buf, test_images_buf);
    dimension_mask(masked_image_buf, 2, n);
    masked_image_buf.host_dirty = true;

    // run pipeline
    tm.start();
    int error = mnist_opencl_pipeline(&masked_image_buf, &output_buf);
    tm.stop();
    if (error) {
      printf ("Halide Error: %d\n\n", error);
      return -1;
    }

    // find predicted label
    uint8_t prediction = 0;
    float *probs = (float *) output_buf.host;
    float prediction_prob = probs[0];
    for(int i = 0; i < output_buf.extent[0]; i++) {
      if(probs[i] > prediction_prob) {
	prediction = i;
	prediction_prob = probs[i];
      }
    }

    // ground truth
    uint8_t label = test_labels[n];
    
    // print and collect results
    printf ("Image %4d: Predicted %2d vs %2d\t\t[%s]\n", n, prediction, label, (prediction==label)?"Pass":"Fail");
    correct += (prediction==label) ? 1 : 0;
  }

  // print results
  int total = num_stop - num_start + 1;
  printf ("Accuracy: %d/%d = %4.2f%%\n", correct, total, float(correct)/total*100);

  delete[] output_buf.host;
  delete[] test_images_buf.host;
  delete[] test_labels;

  return 0;
}

