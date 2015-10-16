#include "cnn.hpp"
#include "layers.hpp"
#include "common.hpp"
#include "Halide.h"
#include <string>
#include <boost/date_time/posix_time/posix_time.hpp>
#include "../../utils/read_mnist.h"
#include "../../utils/read_params.h"
#include "buffer_io.hpp"
#include <vector>
#include <utility>
#include <gflags/gflags.h>

using namespace Halide;
using namespace std;

DEFINE_string(param_files, "", "parameters for the cnn layers");
DEFINE_string(target, "cpu", "target to build for (cpu, opencl, cuda)");
DEFINE_int32(batch, 64, "batch size");
DEFINE_int32(iterations, 156, "number of iterations");

// This main function creates the lenet cnn pipeline and aot compiles it
int main(int argc, char **argv) {
  // setup gflags
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // parse param_files flags to generate paths
  vector<string> param_files = split(FLAGS_param_files, ',');

  // parse param files
  pair<Image<float>, Image<float>> param0 = 
    read_dat_params(param_files[0], "param0", 0);
  pair<Image<float>, Image<float>> param1 =
    read_dat_params(param_files[1], "param1", 0);
  pair<Image<float>, Image<float>> param2 =
    read_dat_params(param_files[2], "param2", 1);
  pair<Image<float>, Image<float>> param3 =
    read_dat_params(param_files[3], "param3", 1);

  // load test data
  int count, rows, cols;
  const int channels = 1;
  uint8_t *test_images = read_mnist_test_images(count, rows, cols); 
  buffer_t test_images_buf{0};
  init_buffer(test_images_buf, cols, rows, 1, count, 1, test_images);

  // load output buffer
  buffer_t output_buf{0};
  init_buffer(output_buf, 10, FLAGS_batch, 0, 0, 
	      sizeof(float),(uint8_t *) (new float[10*FLAGS_batch]));
  Buffer output(Float(32), &output_buf);

  // setup initial data dimensions
  vector<Expr> data_dim {rows, cols, channels};

  // setup input function
  Var x("x"), y("y"), c("c"), n("n");
  ImageParam input_image(type_of<uint8_t>(), 4);
  vector<Argument> args(1);
  args[0] = input_image;
  Func input("input");
  input(x, y, c, n) = cast<float>(input_image(x, y, c, n)) / 255;

  // setup pipeline functions
  Func conv0 = 
    convolutional_layer<float>(input, data_dim, param0.first,
			param0.second, "conv0");
  Func maxpool1 = 
    maxpool_layer(conv0, data_dim, "maxpool1");
  Func conv2 =
    convolutional_layer<float>(maxpool1, data_dim, param1.first,
			param1.second, "conv2");
  Func maxpool3 =
    maxpool_layer(conv2, data_dim, "maxpool3");
  Func ip4 =
    innerproduct_layer<float>(maxpool3, data_dim, param2.first,
		       param2.second, "ip4");
  Func relu5 = 
    relu_layer(ip4, data_dim, "relu5");
  Func ip6 =
    innerproduct_layer<float>(relu5, data_dim, param3.first,
		       param3.second, "ip6");
  Func softmax7 =
    softmax_layer(ip6, data_dim, "softmax7");

  // schedule and set target
  Target target;
  if(!FLAGS_target.compare("cpu")) {
    // schedule for the host cpu
    conv0.compute_root().vectorize(x, 8).parallel(n);
    maxpool1.compute_root().vectorize(x, 8).parallel(n);
    conv2.compute_root().vectorize(x, 8).parallel(n);
    maxpool3.compute_root().vectorize(x, 8).parallel(n);
    ip4.compute_root().vectorize(x, 8).parallel(n);
    relu5.compute_root().vectorize(x, 8).parallel(n);
    ip6.compute_root().vectorize(x, 8).parallel(n);
    softmax7.compute_root().vectorize(x, 8).parallel(n);
    target = get_jit_target_from_environment();
  } else {
    // schedule for gpu
    conv0.compute_root().vectorize(x, 8); 
    maxpool1.compute_root().vectorize(x, 8);
    conv2.compute_root().vectorize(x, 8);  
    maxpool3.compute_root().vectorize(x, 8);
    ip4.compute_root().vectorize(x, 8);
    relu5.compute_root().vectorize(x, 8);
    ip6.compute_root().vectorize(x, 8);
    softmax7.compute_root().vectorize(x, 8);

    // get gpu target
    target.os = Target::Linux;
    target.arch = Target::X86;
    target.bits = 64;
    target.set_feature(Target::SSE41); 
    if(!FLAGS_target.compare("opencl"))
      target.set_feature(Target::OpenCL);
    else if(!FLAGS_target.compare("cuda")) {
      target.set_feature(Target::CUDA);
      target.set_feature(Target::CUDACapability35); 
    } else {
      printf("Please specify a proper target (cpu, opencl, or cuda)\n");
      exit(1);
    }
  }

  // computing time is the amount of time spent on the pipeline.
  // it should be noted that caffe benchmarks are based on computing time
  float computing_time = 0;
  float total_time = 0;
  boost::posix_time::ptime computing_start;
  boost::posix_time::ptime computing_end;
  boost::posix_time::ptime total_end;
  boost::posix_time::ptime total_start =
    boost::posix_time::microsec_clock::local_time();
    
  // jit compile
  softmax7.infer_input_bounds(FLAGS_batch, 10, 0, 0);
  softmax7.compile_jit(target);

  for(int n = 0; n < FLAGS_iterations; n++) {
    // prepare input
    buffer_t masked_image_buf = test_images_buf;
    dimension_mask(masked_image_buf, 3, n*FLAGS_batch, FLAGS_batch);
      
    // set input
    Buffer input(UInt(8), &masked_image_buf);
    input_image.set(input);

    // run pipeline
    computing_start = boost::posix_time::microsec_clock::local_time();
    softmax7.realize(output);
    computing_end = boost::posix_time::microsec_clock::local_time();
    computing_time += (computing_end - computing_start).total_milliseconds();      
  }
    
  // print performance results
  total_end = boost::posix_time::microsec_clock::local_time();
  total_time = (total_end - total_start).total_milliseconds();
  printf("Total time: %g\n", total_time);
  printf("Computing time: %g\n", computing_time);

  delete[] output_buf.host;
  delete[] test_images;

  return 0;
}
