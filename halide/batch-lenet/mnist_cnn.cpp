#include "cnn.hpp"
#include "layers.hpp"
#include "common.hpp"
#include "Halide.h"
#include <string>
#include <vector>
#include <utility>
#include <gflags/gflags.h>

using namespace Halide;
using namespace std;

DEFINE_string(pipeline_name, "", "name of the generated pipeline");
DEFINE_string(param_files, "", "parameters for the cnn layers");
DEFINE_string(target, "cpu", "target to build for (cpu, opencl, cuda)");
DEFINE_int32(input_x, 28, "input width");
DEFINE_int32(input_y, 28, "input height");
DEFINE_int32(input_c, 1, "input depth");

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

  // setup initial data dimensions
  vector<Expr> data_dim {FLAGS_input_x, FLAGS_input_y, FLAGS_input_c};

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

  // schedule and aot compile
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

    // aot compile for the host cpu
    softmax7.compile_to_file(FLAGS_pipeline_name, args);
    printf("Successfully compiled pipeline\n");
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
    Target target;
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
    
    // aot compile for gpu
    softmax7.compile_to_file(FLAGS_pipeline_name, args, target);
    printf("Successfully compiled pipeline\n");
  }
  
  return 0;
}
