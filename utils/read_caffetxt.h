#ifndef READ_CAFFETXT_H
#define READ_CAFFETXT_H
#include <stdio.h>
#include <stdint.h>
#include <vector>
#include <utility>
#include <fstream>
#include <string>
#include <sstream>

typedef std::pair<std::string, uint8_t> LabeledName;
std::vector<LabeledName> read_caffe_txt(std::string txt_path, std::string dir_path,
					int num_start, int num_stop) {
  // open a stream and create a vector to store filename-label pairs
  std::fstream stream;
  stream.open(txt_path, std::fstream::in);
  std::vector<LabeledName> image_vector;

  std::string line;
  int linum = 0; // the current line
  while(getline(stream, line)) {
    // check that the current line is within the specified range
    if(linum < num_start) {
      linum++;
      continue;
    }
    if(linum > num_stop) break;

    // parse current line
    std::vector<std::string> string_vec;
    std::stringstream ss(line);
    std::string item;
    while(std::getline(ss, item, ' '))
      string_vec.push_back(item);
    if(string_vec.size() != 2) break;

    // parse and store the filename-label pair
    image_vector.push_back(std::make_pair(dir_path + "/" + string_vec[0], stoi(string_vec[1])));

    linum++;
  }

  // check that the desired number of filename-label pairs were retrieved
  if(linum <= num_stop) {
    std::fprintf(stderr, "Could not load enough testing images\n");
    exit(1);
  }

  stream.close();
  return image_vector;
}

#endif
