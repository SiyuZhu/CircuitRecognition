#include "layers.hpp"
#include "common.hpp"

// TODO not completely certain that the correct softmax algorithm was used

namespace latte {
  SoftmaxLossLayer::SoftmaxLossLayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = false;
  }

  bool SoftmaxLossLayer::init() {
    if(initialized) return true;
    initialized = true;
    return initialized;
  }

  void SoftmaxLossLayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;
    CHECK(init()) << "Could not build tree because " << this->name()
		  << " could not be initialized." << endl;

    map<string, BoundedFunc>::iterator data_it = 
      func_tree->find(layer_param_.bottom(0));
    CHECK(data_it != func_tree->end()) << "Could not find " <<
      layer_param_.bottom(0) << " in the func tree." << endl;

    map<string, BoundedFunc>::iterator label_it =
      func_tree->find(layer_param_.bottom(1));
    CHECK(label_it != func_tree->end()) << "Could not find " <<
      layer_param_.bottom(1) << " in the func tree." << endl;

    Func& data_func = data_it->second.first;
    array<int, 4> data_dim = data_it->second.second;
    //    Func& label_func = label_it->second.first;
    array<int, 4> labal_dim = label_it->second.second;

    Func softmax_producer(layer_param_.top(0) + "_softmax_producer"),
      softmax(layer_param_.top(0) + "_softmax");
    softmax_producer(x, y, c, n) = exp(data_func(x, y, c, n));
    RDom r(0, data_dim[2]);
    softmax(x, y, c, n) = softmax_producer(x, y, c, n) / 
      sum(softmax_producer(x, y, r, n));

    Func loss(layer_param_.top(0));
    RDom r0(0, data_dim[0], 0, data_dim[1], 0, data_dim[2], 0, data_dim[3]);
    loss(x, y, c, n) = sum(-Halide::log(softmax(r0.x, r0.y, r0.z, r0.w)));

    array<int, 4> loss_dims = {{1, 1, 1, 1}};
    func_tree->insert(make_pair(layer_param_.top(0),
				make_pair(loss, loss_dims)));

    loss.compute_root().parallel(n);

    LOG(INFO) << layer_param_.top(0) << ": {" << loss_dims[0] << ", "
	      << loss_dims[1] << ", " << loss_dims[2] << ", " << loss_dims[3]
	      << "}" << endl;

    LOG(INFO) << "Completed building " << this->name() << endl;
  }

  bool SoftmaxLossLayer::copy_trained_layer(const LayerParameter& param) {
    return false;
  }

}
