#include "layers.hpp"
#include "common.hpp"

// TODO support for multiple strides, kernels...

namespace latte {
  ConvolutionLayer::ConvolutionLayer(const LayerParameter& param) {
    layer_param_ = param;
    initialized = false;
    init();
  }

  ConvolutionLayer::~ConvolutionLayer() {
    if(weights_.defined())
      weights_ = ImageParam();
    if(bias_.defined())
      bias_ = ImageParam();
    if(weights_host_ != NULL) {
      delete[] weights_host_;
      weights_host_ = NULL;
    }
    if(bias_host_ != NULL) {
      delete[] bias_host_;
      bias_host_ = NULL;
    }
  }


  bool ConvolutionLayer::init() {
    if(initialized) return true;
    if((layer_param_.blobs_size() < 1) ||
       (layer_param_.convolution_param().bias_term() && 
	layer_param_.blobs_size() < 2)) return false;

    LOG(INFO) << "Starting to initialize " << this->name() << endl;

    const BlobProto& weights_blob = layer_param_.blobs(0);
    LOG(INFO) << "CONV WEIGHTS: " << endl << weights_blob.width() << endl
	      << weights_blob.height() << endl << weights_blob.channels() << endl
	      << weights_blob.num() << endl;

    weights_host_ = new float[weights_blob.width()*weights_blob.height()*
			      weights_blob.channels()*weights_blob.num()];
    Buffer weights_buf(type_of<float>(), weights_blob.width(), weights_blob.height(),
		      weights_blob.channels(), weights_blob.num(), 
		      (uint8_t*) (weights_host_));
    for(int x = 0, x_end = weights_blob.width(); x < x_end; x++) {
      for(int y = 0, y_end = weights_blob.height(); y < y_end; y++) {
	for(int c = 0, c_end = weights_blob.channels(); c < c_end; c++) {
	  for(int n = 0, n_end = weights_blob.num(); n < n_end; n++) {
	    int index = x + y*weights_blob.width() + c*weights_blob.width()*
	      weights_blob.height() + n*weights_blob.width()*
	      weights_blob.height()*weights_blob.channels();
	    weights_host_[index] = weights_blob.data(index);
	  }
	}
      }
    }
    
    if(!weights_.defined())
      weights_ = ImageParam(weights_buf.type(), 4);
    weights_.set(weights_buf);

    if(layer_param_.convolution_param().bias_term()) {
      const BlobProto bias_blob = layer_param_.blobs(1);
      bias_host_ = new float[bias_blob.width()*bias_blob.height()*
			     bias_blob.channels()*bias_blob.num()];
      Buffer bias_buf(type_of<float>(), bias_blob.width(), bias_blob.height(),
		      bias_blob.channels(), bias_blob.num(),(uint8_t*) bias_host_);

      LOG(INFO) << "CONV BIAS" << endl << bias_blob.width() << endl
		<< bias_blob.height() << endl << bias_blob.channels() << endl 
		<< bias_blob.num() << endl; 
      int bias_size = bias_blob.width()*bias_blob.height()*bias_blob.channels()*
	bias_blob.num();
      for(int i = 0; i < bias_size; i++) {
	bias_host_[i] = bias_blob.data(i);
      }
      if(!bias_.defined())
	bias_ = ImageParam(bias_buf.type(), 4);
      bias_.set(bias_buf);
    }

    LOG(INFO) << "Completed initializing " << this->name() << endl;

    initialized = true;
    return initialized;
  }

  bool ConvolutionLayer::copy_trained_layer(const LayerParameter& param) {
    if(param.blobs_size() < 1) return true;
    if(layer_param_.blobs_size() < 1)
      layer_param_.add_blobs()->CopyFrom(param.blobs(0));
    else
      layer_param_.mutable_blobs(0)->CopyFrom(param.blobs(0));
    if(layer_param_.convolution_param().bias_term() && 
       param.convolution_param().bias_term()) {
      if(layer_param_.blobs_size() < 2)
	layer_param_.add_blobs()->CopyFrom(param.blobs(1));
      else
	layer_param_.mutable_blobs(1)->CopyFrom(param.blobs(1));
    }
    initialized = false;
    init();
    return true;
  }

  void ConvolutionLayer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(INFO) << "Starting to build " << this->name() << endl;
    CHECK(init()) << "Cannot build tree because " << this->name()
		  << " could not be initialized." << endl;
      
    map<string, BoundedFunc>::iterator input_it = 
      func_tree->find(layer_param_.bottom(0));
    CHECK(input_it != func_tree->end()) << "Could not find " <<
      layer_param_.bottom(0) << " in the function tree" << endl;

    Func& input_func = (input_it->second).first;
    array<int, 4> dims = (input_it->second).second;

    Func conv(layer_param_.top(0));
    RDom r(0, weights_.extent(0), 0, weights_.extent(1), 0, weights_.extent(2));
    conv(x, y, c, n) = sum(weights_(weights_.extent(0)-1-r.x,
				    weights_.extent(1)-1-r.y, r.z, c) *
			   input_func(x+weights_.extent(0)-1-r.x,
				      y+weights_.extent(1)-1-r.y, r.z, n));
    if(layer_param_.convolution_param().bias_term())
      conv(x, y, c, n) += bias_(c, 0, 0, 0);

    dims[2] = layer_param_.blobs(0).num();
    const ConvolutionParameter& conv_param = layer_param_.convolution_param();
    dims[1] = (dims[1] + 2*conv_param.pad() - layer_param_.blobs(0).height())/
	       conv_param.stride() + 1;
    dims[0] = (dims[0] + 2*conv_param.pad() - layer_param_.blobs(0).width())/
	       conv_param.stride() + 1;

    conv.compute_root().vectorize(x, 8).parallel(n);
    func_tree->insert(make_pair(layer_param_.top(0),
				make_pair(conv, dims)));

    LOG(INFO) << layer_param_.top(0) << ": {" << dims[0] << ", "
	      << dims[1] << ", " << dims[2] << ", " << dims[3] << "}" << endl;

    LOG(INFO) << "Completed building " << this->name() << endl;
  }
}
