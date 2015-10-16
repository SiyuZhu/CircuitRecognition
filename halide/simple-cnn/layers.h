#ifndef LAYERS_H
#define LAYERS_H

#include <Halide.h>
using namespace Halide;
#include "params.h"

#define DEBUG_CONV_OUTPUT 0x1
#define DEBUG_POOL_OUTPUT 0x2
#define DEBUG_BIAS_OUTPUT 0x4
#define DEBUG_HIDDEN_OUTPUT 0x8
#define DEBUG_LAST_OUTPUT 0x10

// Defines a single convolutional layer
//
// This layer will first do a convolution using the kernels,
// then add the biases
// then apply ReLU to the result
//
template <typename T>
Func convolutional_layer (
    Func input,
    Func kernel,
    std::string name,
    int input_x, int input_y, 
    Parameters p,
    int& output_x, int& output_y,
    int flags=0
  );

// Defines a single maxpooling layer
//
template <typename T>
Func maxpool_layer (
    Func input,
    Func bias,
    std::string name,
    int input_x, int input_y, int input_num,
    int pool_mask_x, int pool_mask_y,
    int& pool_x, int& pool_y,
    int flags=0
  );

// Defines a fully-connected hidden layer
//
// This layer will multiply the input with weight, then add the biases,
// perform tanh of the result
template <typename T>
Func hidden_layer(
     Func input,
     Func weight,
     Func bias,
     std::string name,
     int width, int height, 
     int depth, int n_out,
     int flags=0
  );
  
//  Defines a fully-connected layer whose previous layer is also fully-connected
//
// This layer will multiply the previous layer with weight, then add the biases,
// perform tanh of the result
template <typename T>
Func dense_layer(
     Func input,
     Func weight,
     Func bias,
     std::string name,
     Parameters p,
     int flags=0
  );
  
// Defines a softmax layer
// This layer will perform softmax on previous layer
template <typename T>
Func softmax_layer(
     Func input,
     int width,
     std::string name
  );
  
#include "layers.cpp"

#endif
