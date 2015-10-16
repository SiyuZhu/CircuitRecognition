#include "layers.hpp"
#include "common.hpp"

namespace latte {
  SoftmaxLayer::SoftmaxLayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = false;
    init();
  }

  bool SoftmaxLayer::init() {
    if(initialized) return true;
    initialized = true;
    return initialized;
  }

  void SoftmaxLayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;
    CHECK(init()) << "Could not build tree because " << this->name()
		  << " could not be initialized." << endl;

    map<string, BoundedFunc>::iterator input_it = 
      func_tree->find(layer_param_.bottom(0));
    CHECK(input_it != func_tree->end()) << "Could not find " <<
      layer_param_.bottom(0) << " in the function tree" << endl;

    Func& input_func = input_it->second.first;
    array<int, 4> dims = input_it->second.second;

    Func producer(layer_param_.top(0) + "_producer");
    producer(x, y, c, n) = exp(input_func(x, y, c, n));

    Func softmax(layer_param_.top(0));
    RDom r(0, dims[2]);
    softmax(x, y, c, n) = producer(x, y, c, n) / sum(producer(x, y, r.x, n));

    func_tree->insert(make_pair(layer_param_.top(0),
				make_pair(softmax, dims)));    

    softmax.compute_root().vectorize(x, 8).parallel(n);

    LOG(INFO) << layer_param_.top(0) << ": {" << dims[0] << ", "
	      << dims[1] << ", " << dims[2] << ", " << dims[3] << "}" << endl;

    LOG(INFO) << "Completed building " << this->name() << endl;
  }

  bool SoftmaxLayer::copy_trained_layer(const LayerParameter& param) {
    return false;
  }
}
