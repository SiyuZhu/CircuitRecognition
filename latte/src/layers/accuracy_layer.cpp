
#include "layers.hpp"
#include "common.hpp"

// TODO top-k accuracy

namespace latte {
  AccuracyLayer::AccuracyLayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = true;
  }

  bool AccuracyLayer::init() {
    return initialized;
  }

  bool AccuracyLayer::copy_trained_layer(const LayerParameter& param) {
    return false;
  }

  void AccuracyLayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;
    string probs_name = layer_param_.bottom(0);
    string labels_name = layer_param_.bottom(1);

    BoundedFunc& probs_bf = (*func_tree)[probs_name];
    Func& probs = probs_bf.first;
    BoundedFunc& labels_bf = (*func_tree)[labels_name];
    Func& labels = labels_bf.first;

    array<int, 4> accuracy_dims = labels_bf.second;
    accuracy_dims[3] = 1;
    
    Func prediction("prediction");
    RDom r(0, probs_bf.second[2]);
    prediction(x, y, c, n) = argmax(probs(x, y, r, n))[0];
    Func accuracy(layer_param_.top(0));
    RDom r0(0, probs_bf.second[3]);
    accuracy(x, y, c, n) = (cast<float>(sum(select(prediction(x, y, c, r0) 
						  == labels(x, y, c, r0), 1, 0)))
			    /probs_bf.second[3]);
    accuracy.compute_root().parallel(n);

    func_tree->insert(make_pair(layer_param_.top(0),
				make_pair(accuracy, accuracy_dims)));

    LOG(INFO) << layer_param_.top(0) << ": {" << accuracy_dims[0] << ", " 
	      << accuracy_dims[1] << ", " << accuracy_dims[2] << ", " 
	      << accuracy_dims[3] << "}" << endl;

    LOG(INFO) << "Completed building " << this->name() << endl;

  }
}
