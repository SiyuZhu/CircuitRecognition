#ifndef TOOLS_COMMON_HPP
#define TOOLS_COMMON_HPP
#include <vector>
#include <boost/filesystem.hpp>

using namespace std;
namespace fs = boost::filesystem;

// Returns a vector with indices 0-4 containing the min, first quartile, median,
// third quartile, and max, respectively, of the elements in values
template <typename T>
vector<T> *get_quartiles(vector<T> * const& values) {
  sort(values->begin(), values->end());
  vector<T> *quartiles = new vector<T>();
  quartiles->push_back((*values)[0]);
  quartiles->push_back((*values)[(int) (values->size()*.25)]);
  quartiles->push_back((*values)[(int) (values->size()*.5)]);
  quartiles->push_back((*values)[(int) (values->size()*.75)]);
  quartiles->push_back((*values)[values->size()-1]);
  return quartiles;
}

vector<string> *split(const string &s, char delim,
		      vector<string> * const& elems);

vector<string> *split(const string &s, char delim);

// Deletes and then recreates given directory
void recreate(const fs::path& dir);

#endif
