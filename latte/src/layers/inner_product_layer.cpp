#include "layers.hpp"
#include "common.hpp"

namespace latte {
  InnerProductLayer::InnerProductLayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = false;
    init();
  }

  InnerProductLayer::~InnerProductLayer() {
    //    if(weights_.defined())
    //      weights_ = ImageParam();
    //    if(bias_.defined())
    //      bias_ = ImageParam();
    if(weights_host_ != NULL) {
      delete[] weights_host_;
      weights_host_ = NULL;
    }
    if(bias_host_ != NULL) {
      delete[] bias_host_;
      bias_host_ = NULL;
    }
  }

  bool InnerProductLayer::init() {
    if(initialized) return true;
    if((layer_param_.blobs_size() == 0) ||
       (layer_param_.inner_product_param().bias_term() &&
	layer_param_.blobs_size() < 2)) return false;

    LOG(INFO) << "Starting to initialize " << this->name() << endl;

    BlobProto weights_blob = layer_param_.blobs(0);
    weights_host_ = new float[weights_blob.width()*weights_blob.height()*
			      weights_blob.channels()*weights_blob.num()];
    Buffer weights_buf(type_of<float>(), 1, 1, weights_blob.width(), 
		       weights_blob.height(), (uint8_t*) weights_host_);

    for(int n = 0, n_end = weights_blob.height(); n < n_end; n++) {
      for(int c = 0, c_end = weights_blob.width(); c < c_end; c++) {
	int index = n + c*n_end;
	weights_host_[index] = weights_blob.data(index);
      }
    }
    if(!weights_.defined())
      weights_ = ImageParam(weights_buf.type(), 4);
    weights_.set(weights_buf);

    if(layer_param_.inner_product_param().bias_term()) {
      BlobProto bias_blob = layer_param_.blobs(1);
      bias_host_ = new float[bias_blob.width()*bias_blob.height()*
			     bias_blob.channels()*bias_blob.num()];
      Buffer bias_buf(type_of<float>(),  bias_blob.num(), bias_blob.channels(),
		      bias_blob.height(), bias_blob.width(), (uint8_t*) bias_host_);
      for(int x = 0, x_end = bias_blob.width(); x < x_end; x++) {
	bias_host_[x] = bias_blob.data(x);
      }
      if(!bias_.defined())
	bias_ = ImageParam(bias_buf.type(), 4);
      bias_.set(bias_buf);
    }

    LOG(INFO) << "Completed initializing " << this->name() << endl;

    initialized = true;
    return initialized;
  }

  void InnerProductLayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;
    CHECK(init()) << "Connot build tree because " << this->name()
		  << " could not be initialized." << endl;

    map<string, BoundedFunc>::iterator input_it = 
      func_tree->find(layer_param_.bottom(0));
    CHECK(input_it != func_tree->end()) << "Could not find" <<
      layer_param_.bottom(0) << " in the function tree." << endl;

    Func& input_func = (input_it->second).first;
    array<int, 4> dims = (input_it->second).second;

    Func ip(layer_param_.top(0));
    RDom r(0, dims[0], 0, dims[1], 0, dims[2]);
    ip(x, y, c, n) = sum(input_func(r.x, r.y, r.z, n) *
			 weights_(0, 0, r.x + r.y*dims[0] +
				  r.z*dims[0]*dims[1], c));
    if(layer_param_.inner_product_param().bias_term())
      ip(x, y, c, n) += bias_(0, 0, 0, c);

    dims[2] = weights_.get().extent(1);
    dims[1] = 1;
    dims[0] = 1;

    ip.compute_root().vectorize(x, 8).parallel(n);

    func_tree->insert(make_pair(layer_param_.top(0),
				make_pair(ip, dims)));
    
    LOG(INFO) << layer_param_.top(0) << ": {" << dims[0] << ", "
	      << dims[1] << ", " << dims[2] << ", " << dims[3] << "}" << endl;
    
    LOG(INFO) << "Completed building " << this->name() << endl;
  }

  bool InnerProductLayer::copy_trained_layer(const LayerParameter& param) {
    if(param.blobs_size() < 1) return true;
    if(layer_param_.blobs_size() < 1)
      layer_param_.add_blobs()->CopyFrom(param.blobs(0));
    else 
      layer_param_.mutable_blobs(0)->CopyFrom(param.blobs(0));
    if(layer_param_.inner_product_param().bias_term() &&
       param.inner_product_param().bias_term()) {
      if(layer_param_.blobs_size() < 2)
	layer_param_.add_blobs()->CopyFrom(param.blobs(1));
      else
	layer_param_.mutable_blobs(1)->CopyFrom(param.blobs(1));
    }
    initialized = false;
    init();
    return true;
  }
}
