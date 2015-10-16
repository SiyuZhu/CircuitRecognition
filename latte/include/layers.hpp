#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "stage.hpp"
#include <boost/shared_ptr.hpp>

namespace latte {
  class Layer: public Stage {
  protected:
    LayerParameter layer_param_;
    bool initialized;

  public:
    virtual ~Layer() {}
    // Defined in the end of this file
    static bool is_valid(const LayerParameter& param);
    // Defined in the end of this file
    static boost::shared_ptr<Layer> get_layer(const LayerParameter& param);

    virtual string name() const;
    virtual vector<string> producer_names() const;
    virtual vector<string> output_names() const;

    virtual bool init() = 0;
    virtual bool copy_trained_layer(const LayerParameter& param) = 0;
    virtual void build_tree(map<string, BoundedFunc>* func_tree) = 0;

    inline void to_proto(LayerParameter *const param) const {
      param->CopyFrom(layer_param_);
    }
    virtual void to_proto(NetParameter& net) const;
  };  
}

namespace latte {
  class AccuracyLayer : public Layer {
  public:
    AccuracyLayer(const LayerParameter& param);
    virtual ~AccuracyLayer() {}
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
  class ConvolutionLayer : public Layer {
  protected:
    float *weights_host_, *bias_host_;
    ImageParam weights_;
    ImageParam bias_;
  public:
    ConvolutionLayer(const LayerParameter& param);
    virtual ~ConvolutionLayer();
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
  class InnerProductLayer : public Layer {
    float *weights_host_, *bias_host_;
    ImageParam weights_;
    ImageParam bias_;
  public:
    InnerProductLayer(const LayerParameter& param);
    // TODO 
    virtual ~InnerProductLayer();
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
  class PoolingLayer : public Layer {
  public:
    PoolingLayer(const LayerParameter& param);
    virtual ~PoolingLayer() {}
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
  class ReLULayer : public Layer {
  public:
    ReLULayer(const LayerParameter& param);
    virtual ~ReLULayer() {}
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
  class SoftmaxLayer : public Layer {
  public:
    SoftmaxLayer(const LayerParameter& param);
    virtual ~SoftmaxLayer() {}
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
  class SoftmaxLossLayer : public Layer {
  public:
    SoftmaxLossLayer(const LayerParameter& param);
    virtual ~SoftmaxLossLayer() {}
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
    virtual bool copy_trained_layer(const LayerParameter& param);
  };
}

#endif
