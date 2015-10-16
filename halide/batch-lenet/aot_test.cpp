#include <stdio.h>
#include "mnist_pipeline.h"
#include "../../utils/read_mnist.h"
#include "buffer_io.hpp"
#include <boost/date_time/posix_time/posix_time.hpp>
#include <vector>

using namespace std;

// uses the mnist pipeline to classify the given images
// the first argument specifies the batch size and the second
// argument specifies the number of iterations
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
  float computing_time = 0;
  float total_time = 0;
  boost::posix_time::ptime computing_start;
  boost::posix_time::ptime computing_end;
  boost::posix_time::ptime total_end;
  boost::posix_time::ptime total_start = 
    boost::posix_time::microsec_clock::local_time();

  // classify test images
  int correct = 0;
  for (int n = 0; n < iterations; ++n) {
    // select the image to classify
    buffer_t masked_image_buf =test_images_buf;
    dimension_mask(masked_image_buf, 3, n*batch_size, batch_size);

    // run pipeline
    computing_start = boost::posix_time::microsec_clock::local_time();
    int error = mnist_pipeline(&masked_image_buf, &output_buf);
    computing_end = boost::posix_time::microsec_clock::local_time();
    computing_time += (computing_end - computing_start).total_milliseconds();      

    if (error) {
      printf ("Halide Error: %d\n\n", error);
      return -1;
    }

    // find predicted label
    vector<uint8_t> predictions(batch_size, 0);
    vector<float> prediction_probs(batch_size, 0);
    float *probs = (float *) output_buf.host;
    for(int i = 0; i < batch_size; i++)
      prediction_probs[i] = probs[output_buf.extent[0]*i];
    for(int i = 0; i < output_buf.extent[0]; i++) {
      for(int j = 0; j < batch_size; j++) {
	if(probs[j*output_buf.extent[0] + i] > prediction_probs[j]) {
	  prediction_probs[j] = probs[j*output_buf.extent[0] + i];
	  predictions[j] = i;
	}
      }
    }

    // print and collect accuracy results
    for(int i = 0; i < batch_size; i++) {
      printf ("Image %4d: Predicted %2d vs %2d\t\t[%s]\n", n*batch_size + i, predictions[i], test_labels[n*batch_size + i], (predictions[i]==test_labels[n*batch_size + i])?"Pass":"Fail");
    correct += (predictions[i]==test_labels[n*batch_size + i]) ? 1 : 0;
    }
  }

  // print performance results
  total_end = boost::posix_time::microsec_clock::local_time();
  total_time = (total_end - total_start).total_milliseconds();
  printf("Total time: %g\n", total_time);
  printf("Computing time: %g\n", computing_time);

  // print accuracy results
  int total = batch_size*iterations;
  printf ("Accuracy: %d/%d = %4.2f%%\n", correct, total, float(correct)/total*100);

  delete[] output_buf.host;
  delete[] test_images;
  delete[] test_labels;

  return 0;
}

