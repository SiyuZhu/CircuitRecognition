#ifndef COMMON_H
#define COMMON_H

#include "../support/image_io.h"

// convert C type to Halide type
template<typename T>
struct Halide_Type_Helper;

template<>
struct Halide_Type_Helper<uint8_t> {
  operator Type() { return Halide::UInt(sizeof(uint8_t)*8); }
};
template<>
struct Halide_Type_Helper<float> {
  operator Type() { return Halide::Float(sizeof(float)*8); }
};
  
template<typename T>
Halide::Type Halide_Type() {
  return Type(Halide_Type_Helper<T>());
}


// This function creates a Halide::Image of the appropriate type out of a C array
// It wraps a Buffer around the array, then wraps an Image around that
template<typename T>
Image<T> Halide_image_from_buffer (int width, int height, T* data, std::string name) {
  return Image<T>( Buffer(Halide_Type<T>(), width, height, 0, 0, (uint8_t*)data, name.c_str()) );
}

template<typename T>
Image<T> Halide_image_from_buffer (int width, int height, int depth, T* data, std::string name) {
  return Image<T>( Buffer(Halide_Type<T>(), width, height, depth, 0, (uint8_t*)data, name.c_str()) );
}

template<typename T>
Image<T> Halide_image_from_buffer (int width, int height, int depth, int count, T* data, std::string name) {
  return Image<T>( Buffer(Halide_Type<T>(), width, height, depth, count, (uint8_t*)data, name.c_str()) );
}

// This function creates a Halide::Func out of the appropriate type out of a C array
template<typename T>
Func Halide_func_from_buffer (int width, int height, int depth, int count, T* data, std::string name) {
  Var x("x"), y("y"), z("z"), k("k");
  Image<T> img = Halide_image_from_buffer(width, height, depth, count, data, name+"_img");
  Func F(name);
  F(x,y,z,k) = cast<T>(img(x,y,z,k));
  return F;
}

template<typename T>
Func Halide_func_from_buffer (int width, int height, int depth, T* data, std::string name) {
  Var x("x"), y("y"), z("z");
  Image<T> img = Halide_image_from_buffer(width, height, depth, data, name+"_img");
  Func F(name);
  F(x,y,z) = cast<T>(img(x,y,z));
  return F;
}


// realize and print the func to a file
template<typename T>
Image<T> print_func (Func f, int width, int height, int depth, int layers, std::string name, uint8_t scale=1) {
  Var x("x"), y("y"), z("z"), k("k");
  Image<T> img(width, height, depth, layers, name+"_img");
  Func g;
  g(x,y,z,k) = cast<T>(f(x,y,z,k)*scale);
  g.realize(img);
  print_image_by_layer(img, name+".dat");
  return img;
}

template<typename T>
Image<T> print_func (Func f, int width, int height, int depth, std::string name, uint8_t scale=1) {
  Var x("x"), y("y"), z("z");
  Image<T> img(width, height, depth, name+"_img");
  Func g;
  g(x,y,z) = cast<T>(f(x,y,z)*scale);
  g.realize(img);
  print_image_by_layer(img, name+".dat");
  return img;
}


#endif
