#include "layers.hpp"

namespace latte {
  string Layer::name() const {return layer_param_.name();}

  vector<string> Layer::producer_names() const {
    vector<string> values;
    for(int i = 0, bottom = layer_param_.bottom_size(); i < bottom; i++)
      values.push_back(layer_param_.bottom(i));
    return values;
  }

  vector<string> Layer::output_names() const {
    vector<string> values;
    for(int i = 0, top = layer_param_.top_size(); i < top; i++) 
      values.push_back(layer_param_.top(i));
    return values;
  }

  void Layer::to_proto(NetParameter& net) const {
    *(net.add_layers()) = layer_param_;
  }

  bool Layer::is_valid(const LayerParameter& param) {
    if(!param.has_type()) return false;
    switch (param.type()) {
    case LayerParameter_LayerType_ACCURACY: return true;
    case LayerParameter_LayerType_CONVOLUTION: return true;
    case LayerParameter_LayerType_INNER_PRODUCT: return true; 
    case LayerParameter_LayerType_POOLING: return true; 
    case LayerParameter_LayerType_RELU: return true;
    case LayerParameter_LayerType_SOFTMAX: return true; 
    case LayerParameter_LayerType_SOFTMAX_LOSS: return true;
    default: return false;
    }
    return true;
  }

  boost::shared_ptr<Layer> Layer::get_layer(const LayerParameter& param) {
    CHECK(Layer::is_valid(param)) << "Cannot generate layer \"" << param.name()
				  << "\"." << endl;
    switch (param.type()) {
    case LayerParameter_LayerType_ACCURACY:
      return boost::shared_ptr<Layer>(new AccuracyLayer(param));
      break;
    case LayerParameter_LayerType_CONVOLUTION:
      return boost::shared_ptr<Layer>(new ConvolutionLayer(param));
      break;
    case LayerParameter_LayerType_INNER_PRODUCT:
      return boost::shared_ptr<Layer>(new InnerProductLayer(param));
      break;
    case LayerParameter_LayerType_POOLING:
      return boost::shared_ptr<Layer>(new PoolingLayer(param));
      break;
    case LayerParameter_LayerType_RELU:
      return boost::shared_ptr<Layer>(new ReLULayer(param));
      break;
    case LayerParameter_LayerType_SOFTMAX: 
      return boost::shared_ptr<Layer>(new SoftmaxLayer(param));
      break;
    case LayerParameter_LayerType_SOFTMAX_LOSS:
      return boost::shared_ptr<Layer>(new SoftmaxLossLayer(param));
      break;
    default: LOG(FATAL) << "Cannot generate layer \"" << param.name()
			<< "\"." << endl;  
    }
    return boost::shared_ptr<Layer>(new AccuracyLayer(param)); // Dummy return
  }
}
