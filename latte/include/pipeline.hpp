#ifndef PIPELINE_HPP
#define PIPELINE_HPP

#include "stage.hpp"
#include "layers.hpp"
#include "producers.hpp"
#include <unordered_set>

namespace latte {
  class Pipeline : public Stage {
    bool completed_tree_;
    NetParameter net_param_;
    map<string, BoundedFunc> func_tree_;
    map<string, boost::shared_ptr<Layer>> layers_;
    map<string, boost::shared_ptr<Producer>> producers_;
    map<string, string> output_to_layer_;
    unordered_set<string> output_names_;
    
  public:
    explicit Pipeline(const NetParameter& param);
    explicit Pipeline(const string param_file);
    virtual ~Pipeline();
    // builds internal pipeline tree if required
    virtual bool init();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);

    // jit compiles Func name in the internal tree
    virtual void* compile_jit(string name, const Target& target =
			      Latte::singleton()->target());
    // realizes Func name onto the given output
    virtual void realize(Buffer& output, string name, 
			 const Target& target =
			 Latte::singleton()->target());
    // TODO
    // virtual void update_producers();

    // TODO calls realize for all each output, and prints results
    // virtual void forward();

    // TODO calls realize for given outputs, and prints results
    // virtual void forward(vector<string> names);

    // output dimension sizes of Func name
    virtual array<int,4> dims(string name) {
      init();
      map<string, BoundedFunc>::iterator it = 
	func_tree_.find(name);
      CHECK(it != func_tree_.end()) << name << "was not a valid function" << endl;
      return it->second.second;
    }

    string name() const {return net_param_.name();}
    vector<string> producer_names() const {
      vector<string> producer_names;
      for(map<string, boost::shared_ptr<Producer>>::const_iterator it
	    = producers_.begin(); it != producers_.end(); it++)
	producer_names.push_back(it->first);
      return producer_names;
    }
    vector<string> output_names() const {
      vector<string> names;
      for(unordered_set<string>::const_iterator it = output_names_.begin(), 
	    it_end = output_names_.end(); it != it_end; it++)
	names.push_back(*it);
      return names;
    }

    bool copy_trained_layers(const NetParameter& param);
    bool copy_trained_layers(const string filename);

    void to_proto(NetParameter& net) const {net.CopyFrom(net_param_);}
    void to_proto(NetParameter *const param) const {
      param->CopyFrom(net_param_);
    }

    static bool state_meets_rule(const NetState& state, const NetStateRule& rule);
  private:
    void filter_pipeline(const NetParameter& param,
			 NetParameter *const param_filtered);

    // creates required producers and layers. determines outputs.
    void initialize(const NetParameter& param);
  };
}
  
#endif
