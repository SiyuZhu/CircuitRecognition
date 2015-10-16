#include "layers.hpp"
#include "common.hpp"

// TODO account for negative slope

namespace latte {
  ReLULayer::ReLULayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = false;
    init();
  }

  bool ReLULayer::init() {
    if(initialized) return true;
    initialized = true;
    return initialized;      
  }

  void ReLULayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;
    CHECK(init()) << "Could not build the function tree because " <<
      this->name() << " could not be initialized" << endl;

    map<string, BoundedFunc>::iterator input_it = 
      func_tree->find(layer_param_.bottom(0));
    CHECK(input_it != func_tree->end()) << "Could not find " <<
      layer_param_.bottom(0) << " in the function tree." << endl;

    Func& input_func = (input_it->second).first;
    array<int, 4> dims = (input_it->second).second;

    Func relu(layer_param_.top(0));
    relu(x, y, c, n) = max(input_func(x, y, c, n), 0);

    relu.compute_root().vectorize(x, 8).parallel(n);
    func_tree->insert(make_pair(layer_param_.top(0),
				make_pair(relu, dims)));

    LOG(INFO) << layer_param_.top(0) << ": {" << dims[0] << ", "
	      << dims[1] << ", " << dims[2] << ", " << dims[3] << "}" << endl;

    LOG(INFO) << "Completed building " << this->name() << endl;
  }

  bool ReLULayer::copy_trained_layer(const LayerParameter& param) {
    return false;
  }
}
