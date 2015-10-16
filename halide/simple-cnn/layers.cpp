#ifndef LAYERS_CPP
#define LAYERS_CPP

#include "layers.h"
#include "convolution.h"
#include "common.h"

using namespace Halide;

template <typename T>
Func convolutional_layer (
    Func input,
    Func kernel,
    Func bias,
    std::string name,
    int input_x, int input_y, 
    Parameters p,
    int& output_x, int& output_y,
    int flags
  )
{

  // perform convolution
  std::string conv_name = name + "_conv";
  Func conv_result = convolve<T>(
                        input, 
                        kernel, 
                        p.width, p.height,
                        p.layers, p.count,
                        conv_name
      );
  
  // calculate resulting size
  output_x = input_x - p.width + 1;
  output_y = input_y - p.height + 1;
  assert (output_x > 0);
  assert (output_y > 0);

  if (flags & DEBUG_CONV_OUTPUT) {
    print_func<T>(conv_result, output_x, output_y, p.count, conv_name);
  }

  // add biases to the conv result
  Var x("x"), y("y"), z("z");
  std::string bias_name = name + "_bias";
  Func bias_result(bias_name);
  bias_result(x,y,z) = conv_result(x,y,z)+bias(0,0,0,z);

  //conv_result.compute_at(bias_result, x);
  //conv_result.vectorize(x, 8);
  //bias_result.compute_root();
  //bias_result.vectorize(x, 8);
  
  if (flags & DEBUG_BIAS_OUTPUT) {
    print_func<T>(bias_result, output_x, output_y, p.count, bias_name);
  }

  return bias_result;
}

template <typename T>
Func maxpool_layer (
    Func input,
    std::string name,
    int input_x, int input_y, int input_num,
    int pool_mask_x, int pool_mask_y,
    int& pool_x, int& pool_y,
    int flags
  )
{
  Var x("x"), y("y"), z("z");
  std::string pool_name = name + "_pool";
  Func pool_result = maxpool<T>(
              input, 
              pool_mask_x, pool_mask_y, 
              input_num, 
              pool_name
          );

  // calculate resulting size
  pool_x = input_x / pool_mask_x;
  pool_y = input_y / pool_mask_y;
  assert (pool_x > 0);
  assert (pool_y > 0);
  
  if (flags & DEBUG_POOL_OUTPUT) {
    print_func<T>(pool_result, pool_x, pool_y, input_num, pool_name);
  }
  
  return pool_result;
}

// generate layer 3 -- hidden layer
// input is a 3D stack of feature maps width x height x depth
// output is 1D n_out array
template <typename T>
Func hidden_layer(
     Func input,
     Func weight,
     Func bias,
     std::string name,
     int width, int height, int depth,
     int n_out,
     int flags
  )
{
  Var x("x"), y("y"), z("z");
  RDom r(0,width, 0,height, 0,depth);//Perhaps, the depth should be restricted to 1?
		
  // each pixel in the result is weighted sum of every input pixel
  // then add bias and perform relu normalization
  std::string hidden_name = name + "_hidden";
  Func hidden_result(hidden_name);
  hidden_result(x, y, z) = Halide::max(
      sum(weight(x, r.x, r.y, r.z) * input(r.x, r.y, r.z)) + bias(0,0,0,x),
      0
    );
	
  if(flags & DEBUG_HIDDEN_OUTPUT){
    print_func<T>(hidden_result, n_out, 1, 1, hidden_name);
  }
  return hidden_result;
}

//generate layer 4
template <typename T>
Func dense_layer(      // width x 1 x 1
     Func input,      // height x 1 x 1
     Func weight,     // width x height x 1 x 1
     Func bias,       // 1 x 1 x 1 x width
     std::string name,
     Parameters p,
     int flags
  )
{
  Var x("x"), y("y"), z("z");
 
  // simply perform dot product of weights and inputs
  RDom r_in(0,p.height);
  Func dense(name);
  dense(x,y,z) = sum(weight(x,r_in.x,y,z) * input(r_in.x, y, z)) + bias(0,0,0,x);

  return dense;
} 

// generate softmax layer
template<typename T>
Func softmax_layer(
     Func input,
     int width,
     std::string name
  )
{
  Var x("x"), y("y"), z("z");
  
  // name_exp performs pointwise exponentiation
  std::string exp_name = name + "_exp";
  Func temp_exp(exp_name);
  temp_exp(x,y,z) = Halide::exp(input(x,y,z));

  // perform softmax calculation
  RDom r_in(0, width);
  std::string softmax_name = name + "_softmax";
  Func softmax(softmax_name);
  softmax(x,y,z) = temp_exp(x,y,z) / sum(temp_exp(r_in.x, 0, 0));
  
  return softmax;
}

#endif
