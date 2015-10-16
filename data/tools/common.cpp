#include "common.hpp"
#include <vector>
#include <boost/filesystem.hpp>
#include <algorithm>
#include <string>
#include <sstream>

using namespace std;
namespace fs = boost::filesystem;

// http://stackoverflow.com/questions/236129/split-a-string-in-c
vector<string> *split(const string &s, char delim,
		      vector<string> * const& elems) {
  stringstream ss(s);
  string item;
  while(std::getline(ss, item, delim))
    elems->push_back(item);
  return elems;
}

// http://stackoverflow.com/questions/236129/split-a-string-in-c
vector<string> *split(const string &s, char delim) {
  vector<string> *elems = new vector<string>;
  split(s, delim, elems);
  return elems;
}

// Deletes and then recreates given directory
void recreate(const fs::path& dir) {
  fs::remove_all(dir);
  fs::create_directory(dir);
  return;
}
