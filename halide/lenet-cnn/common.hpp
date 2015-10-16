#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <string>

using namespace std;

// splits the string using the given delimiter, and pushes
// each of the tokens to the back of the elems vector
void split(string s, char delim, vector<string> &elems);

// returns a vector containing the string s split using
// the given delimiter
vector<string> split(string s, char delim);

#endif
