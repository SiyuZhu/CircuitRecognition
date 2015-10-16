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
#include "../../utils/timer.h"

// create an array of fake data
/*template<typename T>
T* create_fake_data (int width, int height, int depth) {
  T* data = new T[width*height*depth];
  // in Halide channels are not interleaved
  for (int k = 0; k < depth; k++)
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        data[k*width*height + i*width + j] = i + k + 1;
  return data;
}*/

template<typename T>
size_t argmax (T* array, size_t size) {
  assert (size >= 1);
  size_t imax = 0;
  T arraymax = array[0];
  for (size_t i = 1; i < size; ++i) {
    if (array[i] > arraymax) {
      imax = i;
      arraymax = array[i];
    }
  }
  return imax;
}


//-------------------------------------------------------------------
// main
//-------------------------------------------------------------------
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

  // typedef struct buffer_t {
  //     uint64_t dev;
  //     uint8_t* host;
  //     int32_t extent[4];
  //     int32_t stride[4];
  //     int32_t min[4];
  //     int32_t elem_size;
  //     bool host_dirty;
  //     bool dev_dirty;
  // } buffer_t;
  //
  // This is how Halide represents input and output images in
  // pre-compiled pipelines. There's a 'host' pointer that points to the
  // start of the image data, some fields that describe how to access
  // pixels, and some fields related to using the GPU that we'll ignore
  // for now (dev, host_dirty, dev_dirty).
  
  // always zero buffers to erase garbage fields
  // define the input buffer
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

  int correct = 0;
  Timer tm("simple-cnn");
  
  for (int n = num_start; n <= num_stop; ++n) {
    // point to the desired image in the data buffer
    input_buf.host = digits + n*cols*rows*sizeof(uint8_t);

    // run pipeline
    tm.start();
    int error = Compiled_Pipeline(&input_buf, &output_buf);
    tm.stop();
    if (error) {
      printf ("Halide Error: %d\n\n", error);
      return -1;
    }

    uint8_t prediction = argmax (output_data, 10);
    uint8_t label = labels[n];

    printf ("Image %4d: Predicted %2d vs %2d\t\t[%s]\n", n, prediction, label, (prediction==label)?"Pass":"Fail");
    correct += (prediction==label) ? 1 : 0;

    /*
    printf ("Input Image:\n");

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; ++j) {
        printf ("%4d ", input_data[i*cols+j]);
      }
      printf ("\n");
    }
    
    printf ("\n");
    printf ("Output Image:\n");

    for (int i = 0; i < 10; i++) {
      printf ("%4.1f ", output_data[i]);
    }
    printf ("\n");
    */
  }

  int total = num_stop - num_start + 1;
  printf ("Accuracy: %d/%d = %4.2f%%\n", correct, total, float(correct)/total*100);
  //printf ("Runtime:\n");

  delete[] digits;
  delete[] labels;

  return 0;
}

