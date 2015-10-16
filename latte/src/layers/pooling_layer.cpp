#include "layers.hpp"
#include "common.hpp"

// TODO Add support for pad_w, kernel_w ...  and Average and stochastic pooling

namespace latte {
  PoolingLayer::PoolingLayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = true;
  }
  
  bool PoolingLayer::init() {
    return initialized;
  }

  bool PoolingLayer::copy_trained_layer(const LayerParameter& param) {
    return false;
  }

  void PoolingLayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;

    map<string, BoundedFunc>::iterator bottom_it = func_tree->find(layer_param_.bottom(0));
    CHECK(bottom_it != func_tree->end()) << "Could not find " <<
      layer_param_.bottom(0) << " in the function tree" << endl;

    Func& input = (bottom_it->second).first;
    array<int,4> dims = (bottom_it->second).second;

    Func pooling(layer_param_.top(0));
    int kernel_size = layer_param_.pooling_param().kernel_size();
    RDom r(0, kernel_size, 0, kernel_size);
    pooling(x, y, c, n) = maximum(input(x * kernel_size + r.x,
				     y * kernel_size + r.y, c, n));
    dims[0] /= kernel_size;
    dims[1] /= kernel_size;
    func_tree->insert(make_pair(layer_param_.top(0), 
				make_pair(pooling, dims)));    

    pooling.compute_root().vectorize(x, 8).parallel(n);

    LOG(INFO) << layer_param_.top(0) << ": {" << dims[0] << ", "
	      << dims[1] << ", " << dims[2] << ", " << dims[3] << "}" << endl;
    
    LOG(INFO) << "Completed building " << this->name() << endl;
  }
}
