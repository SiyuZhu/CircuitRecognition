#include <Halide.h>
#include <stdio.h>
#include "gates_pipeline.h"
#include "../../utils/timer.h"
#include "buffer_io.hpp"
#include "../../utils/read_caffetxt.h"
#include <utility>

using namespace Halide;
using namespace std;

#define GATES_DIR "../../data/gates/test_data"
#define GATES_TXT "../../data/gates/test_data/test.txt" 

// uses the gates pipeline to classify the given images
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
  vector<LabeledName> test_data = read_caffe_txt(GATES_TXT, GATES_DIR,
						 num_start, num_stop);
  // load output buffer
  buffer_t output_buf{0};
  init_buffer(output_buf, 10, 0, 0, 0, sizeof(float), (uint8_t *) (new float[10]));

  // classify test images
  int correct = 0;
  Timer tm("lenet-cnn");
  buffer_t test_image{0};
  for (int n = 0; n <= num_stop-num_start; ++n) {
    // select the image to classify
    printf("starting iteration\n");
    load_buffer(test_data[n].first, test_image);

    // run pipeline
    tm.start();
    int error = gates_pipeline(&test_image, &output_buf);
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
    uint8_t label = test_data[n].second;

    // print and collect results
    printf ("Image %4d: Predicted %2d vs %2d\t\t[%s]\n", n+num_start, prediction, label, (prediction==label)?"Pass":"Fail");
    correct += (prediction==label) ? 1 : 0;

    test_image.host = NULL;
    printf("deleting data\n");
    printf("deleted data\n");
  }

  // print results
  int total = num_stop - num_start + 1;
  printf ("Accuracy: %d/%d = %4.2f%%\n", correct, total, float(correct)/total*100);

  delete[] output_buf.host;

  return 0;
}

