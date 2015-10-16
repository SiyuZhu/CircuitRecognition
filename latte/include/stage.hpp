#ifndef STAGE_HPP
#define STAGE_HPP

#include "Halide.h"
#include "caffe.pb.h"
#include "common.hpp"
#include <glog/logging.h>
#include <string>
#include <vector>
#include <array>
#include <utility>
#include <map>

using namespace Halide;
using namespace caffe;
using namespace std;

typedef pair<Func, array<int, 4>> BoundedFunc;

namespace latte {
  class Stage {

  public:
    virtual ~Stage() {}
    virtual string name() const = 0;
    // Prepares the stage to be realized
    virtual bool init() = 0;
    virtual vector<string> producer_names() const = 0;
    virtual vector<string> output_names() const = 0;
    // Inserts functions for all contained intermediates and outputs into
    // the given tree.
    virtual void build_tree(map<string, BoundedFunc>* func_tree) = 0;
    virtual void to_proto(NetParameter& net) const = 0;
  };
}

#endif
