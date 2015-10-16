#include "common.hpp"
#include <vector>
#include <string>
#include <sstream>

using namespace std;

// http://stackoverflow.com/questions/236129/split-a-string-in-c
void split(string s, char delim,
	   vector<string> &elems) {
  stringstream ss(s);
  string item;
  while(std::getline(ss, item, delim))
    elems.push_back(item);
}

// http://stackoverflow.com/questions/236129/split-a-string-in-c
vector<string> split(string s, char delim) {
  vector<string> elems;
  split(s, delim, elems);
  return elems;
}
