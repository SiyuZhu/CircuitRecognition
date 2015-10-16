#ifndef READ_PARAMS_H
#define READ_PARAMS_H

#include <string>
#include <sstream>
#include <fstream>
#include <stdio.h>

#define DEFAULT_DIR "../../data/simple_cnn_params/"

void read_params (std::string filename, float* &weights, float* &biases, int &numin, int &numout, int &rows, int &cols, int flag) {
  // try without and with the default dir
  std::fstream fs;
  fs.open(filename.c_str(), std::fstream::in);
  if (!fs.is_open()) {
    std::string filename2 = std::string(DEFAULT_DIR) + filename;
    fs.open(filename2.c_str(), std::fstream::in);
  }
  if (!fs.is_open()) {
    fprintf (stderr, "Cannot open file %s\n", filename.c_str());
    exit(1);
  }

  std::string buffer;
  std::stringstream ss;
  std::string s;

  // search for the line "weights"
  while (!fs.eof()) {
    std::getline(fs, buffer);
    std::size_t found = buffer.find("weights");
    if (found != std::string::npos)
      break;
  }
  if (fs.eof()) {
    fprintf (stderr, "Cannot find weights\n");
    exit(1);
  }

  // extract dimensions
  std::getline(fs, buffer);
  ss.str(buffer);

  //Last two layers
  if(flag == 1){
    assert( std::getline(ss, s, ',') );
    numin = atoi(s.c_str());
    assert( std::getline(ss, s, ',') );
    numout = atoi(s.c_str());
    rows = 1;
    cols = 1;
  }
  //First two layers
  else {
    assert( std::getline(ss, s, ',') );
    numout = atoi(s.c_str());
    assert( std::getline(ss, s, ',') );
    numin = atoi(s.c_str());
    assert( std::getline(ss, s, ',') );
    rows = atoi(s.c_str());
    assert( std::getline(ss, s, ',') );
    cols = atoi(s.c_str());
  }


  // read weights
  weights = new float[numin*numout*rows*cols];
  float* wptr = weights;
  if(flag == 1){
    for (int i = 0; i < numin; ++i) {
    std::getline(fs, buffer);
    ss.str(buffer);
    for (int j = 0; j < numout; ++j) {
      assert( std::getline(ss, s, ',') );
      *wptr = atof(s.c_str());
      wptr++;
    }
   }
  }
  
  else
  for (int i = 0; i < numin*numout; ++i) {
    std::getline(fs, buffer);
    ss.str(buffer);
    for (int j = 0; j < rows*cols; ++j) {
      assert( std::getline(ss, s, ',') );
      *wptr = atof(s.c_str());
      wptr++;
    }
  }

  // search for the line "biases"
  while (!fs.eof()) {
    std::getline(fs, buffer);
    std::size_t found = buffer.find("biases");
    if (found != std::string::npos)
      break;
  }
  if (fs.eof()) {
    fprintf (stderr, "Cannot find weights\n");
    exit(1);
  }

  std::getline(fs, buffer);
  ss.str(buffer);
  assert( std::getline(ss, s, ',') );
    assert (numout == atoi(s.c_str()));

  // read biases
  biases = new float[numout];
  float* bptr = biases;
  std::getline(fs, buffer);
  ss.str(buffer);
  for (int i = 0; i < numout; ++i) {
    assert( std::getline(ss, s, ',') );
    *bptr = atof(s.c_str());
    bptr++;
  }

  fs.close();
}

#endif
