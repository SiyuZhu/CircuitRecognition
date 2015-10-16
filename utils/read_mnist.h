#ifndef READ_MNIST_H
#define READ_MNIST_H
#include <fstream>

#define MNIST_TEST_IMAGES "../../data/mnist/t10k-images-idx3-ubyte"
#define MNIST_TEST_LABELS "../../data/mnist/t10k-labels-idx1-ubyte"

// mnist bytes are in big endian, so most like we need to convert

// read mnist test images
uint8_t* read_mnist_test_images (int &count, int &rows, int &cols) {
  std::fstream fs;
  fs.open(MNIST_TEST_IMAGES, std::fstream::in);
  if (!fs.is_open()) {
    fprintf (stderr, "could not open file %s\n", MNIST_TEST_IMAGES);
    exit(1);
  }

  // check magic number
  int magicn;
  fs.read((char*)&magicn, 4);
  if ( __builtin_bswap32(magicn) != 2051) {
    fprintf (stderr, "mnist data magic number does not match\n");
    exit(1);
  }

  // read image parameters
  fs.read((char*)&count, 4);
  count = __builtin_bswap32(count);
  fs.read((char*)&rows, 4);
  rows = __builtin_bswap32(rows);
  fs.read((char*)&cols, 4);
  cols = __builtin_bswap32(cols);
  // read data
  uint8_t* data = new uint8_t[count*rows*cols];
  fs.read((char*)data, count*rows*cols);

  fs.close();
  return data;
}

// read mnist test labels
uint8_t* read_mnist_test_labels (int &count) {
  std::fstream fs;
  fs.open(MNIST_TEST_LABELS, std::fstream::in);
  if (!fs.is_open()) {
    fprintf (stderr, "could not open file %s\n", MNIST_TEST_LABELS);
    exit(1);
  }

  // check magic number
  int magicn;
  fs.read((char*)&magicn, 4);
  if ( __builtin_bswap32(magicn) != 2049) {
    fprintf (stderr, "mnist data magic number does not match\n");
    exit(1);
  }

  // read image parameters
  fs.read((char*)&count, 4);
  count = __builtin_bswap32(count);
  // read data
  uint8_t* data = new uint8_t[count];
  fs.read((char*)data, count);

  fs.close();
  return data;
}

#endif
