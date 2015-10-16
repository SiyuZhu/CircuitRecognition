#include "pipeline.hpp"
#include "upgrade_proto.hpp"

using namespace latte;

void Pipeline::initialize(const NetParameter& param) {
  // generate filtered net parameters 
  filter_pipeline(param, &net_param_); 
  LOG(INFO) << "Initializing net from parameters: " << endl
	    << net_param_.DebugString();
  // build inputs
  for(int input_id = 0; input_id < net_param_.input_size(); input_id++) {
    boost::shared_ptr<Producer> input_producer(new InputProducer(net_param_.input(input_id), net_param_.input_dim(4*input_id), net_param_.input_dim(4*input_id+1), net_param_.input_dim(4*input_id+2), net_param_.input_dim(4*input_id+3)));
    producers_.insert(make_pair(net_param_.input(input_id), input_producer));
    LOG(INFO) << "Created input producer " << net_param_.input(input_id) << endl;
  }
  unordered_set<string> missing_dependencies;
  // collect other pipeline stages
  for(int layer_id = 0; layer_id < net_param_.layers_size(); layer_id++) {
    const LayerParameter &layer_param = net_param_.layers(layer_id);
    vector<string> outputs;
    vector<string> producers;
    string name;
    if (BaseProducer::is_valid(layer_param)) {
      boost::shared_ptr<Producer> data_producer = 
	BaseProducer::get_producer(layer_param);
      producers_.insert(make_pair(layer_param.name(), data_producer));
      outputs = data_producer->output_names();
      producers = data_producer->producer_names();
      name = data_producer->name();
      LOG(INFO) << "Created data producer " << name << endl;
    } else if (Layer::is_valid(layer_param)) {
      boost::shared_ptr<Layer> layer = Layer::get_layer(layer_param);
      layers_.insert(make_pair(layer_param.name(), layer));
      outputs = layer->output_names();
      producers = layer->producer_names();
      name = layer->name();
      LOG(INFO) << "Created layer " << name << endl;
    } else {
      LOG(FATAL) << "Layer " << layer_param.name()
		 << " was not properly defined" << endl;
    }
    // Add to output_names_ and output_to_layer_
    for(vector<string>::iterator it = outputs.begin(), it_end = outputs.end();
	it != it_end; it++) {
      unordered_set<string>::iterator findit = missing_dependencies.find(*it);
      if (findit != missing_dependencies.end())
	missing_dependencies.erase(findit);
      else
	output_names_.insert(*it);
      output_to_layer_.insert(make_pair(*it, name));
    }
    for(vector<string>::iterator it = producers.begin(), 
	  it_end = producers.end(); it != it_end; it++) {
      map<string, string>::iterator findit = output_to_layer_.find(*it);
      if(findit == output_to_layer_.end())
	missing_dependencies.insert(*it);
      else {
	unordered_set<string>::iterator finduit = output_names_.find(*it);
	if(finduit != output_names_.end())
	  output_names_.erase(*it);
      }
    }    
  }
}

Pipeline::Pipeline(const NetParameter& param) {
  completed_tree_ = false;
  initialize(param);
}

Pipeline::Pipeline(const string param_file) {
  NetParameter param;
  ReadNetParamsFromTextFileOrDie(param_file, &param);
  completed_tree_ = false;
  initialize(param);
}

Pipeline::~Pipeline() {
  layers_.clear();
  producers_.clear();
  func_tree_.clear();
  output_to_layer_.clear();
}

bool Pipeline::init() {
  if(completed_tree_) return true;
  // init producers
  LOG(INFO) << "Starting to initialize " << this->name() << endl;
  for(map<string, boost::shared_ptr<Producer>>::iterator pit = producers_.begin(),
	pit_end = producers_.end(); pit != pit_end; pit++)
    if (!(pit->second)->init()) return false;
  LOG(INFO) << "Done initializing all producers" << endl;
  // init layers
  for(map<string, boost::shared_ptr<Layer>>::iterator lit = layers_.begin(),
	lit_end = layers_.end(); lit != lit_end; lit++)
    if(!(lit->second)->init()) return false;
  LOG(INFO) << "Done initializing all layers" << endl;  
  build_tree(&func_tree_);
  LOG(INFO) << "Completed initializing " << this->name() << endl;
  completed_tree_ = true;
  return completed_tree_;
}

void Pipeline::build_tree(map<string, BoundedFunc>* func_tree) {
  // build producers
  for(map<string, boost::shared_ptr<Producer>>::iterator pit = producers_.begin(),
	pit_end = producers_.end(); pit != pit_end; pit++)
    (pit->second)->build_tree(func_tree);
  // build layers
  stack<string> build_list;
  for(map<string, boost::shared_ptr<Layer>>::iterator lit = layers_.begin(),
	lit_end = layers_.end(); lit != lit_end; lit++) {
    // check if layer was already built
    vector<string> layer_outputs = (lit->second)->output_names();
    bool built = true;
    for(vector<string>::iterator loit = layer_outputs.begin(), 
	  loit_end = layer_outputs.end(); loit != loit_end; loit++) {
      map<string, BoundedFunc>::iterator find_loit = 
	func_tree->find(*loit);
      if(find_loit == func_tree->end()) {
	built = false;
	continue;
      }
    }
    if(built) continue;
    LOG(INFO) << "Building layer " << lit->first << endl;
    // build the layer
    build_list.push(lit->first);
    while(!build_list.empty()) {
      string layer = build_list.top();
      // get all of the layer's producers
      map<string, boost::shared_ptr<Layer>>::iterator find_lay = layers_.find(layer);
      vector<string> producers = (find_lay->second)->producer_names();
      // check if each producer was made
      for(vector<string>::iterator pit = producers.begin(),
	    pit_end = producers.end(); pit != pit_end; pit++) {
	map<string, BoundedFunc>::iterator find_pit = func_tree->find(*pit);
	// if the producer was not made, add its layer to the build list
	if(find_pit == func_tree->end()) {
	  map<string, string>::iterator oit = 
	    output_to_layer_.find(*pit);
	  build_list.push(oit->second);
	}
      }
      // if no additional layer dependencies were added, build the layer
      if(!layer.compare(build_list.top())) {
	(find_lay->second)->build_tree(func_tree);
	build_list.pop();
      }
    }
  }
  return;
}

void* Pipeline::compile_jit(string name, const Target& target) {
  CHECK(init()) << "Could not initialize " << this->name() << endl;
  unordered_set<string>::iterator it = output_names_.find(name);
  CHECK(it == output_names_.end()) << name << " is not an output of "
				    << this->name() << endl;
  return func_tree_[name].first.compile_jit(target);
}

void Pipeline::realize(Buffer& output, string name, const Target& target) {
  CHECK(init()) << "Could not initialize " << this->name() << endl;
  unordered_set<string>::iterator it = output_names_.find(name);
  CHECK(it != output_names_.end()) << name << " is not an output of "
				    << this->name() << endl;
  BoundedFunc func = func_tree_[name];
  func.first.realize(output, target);
}

bool Pipeline::copy_trained_layers(const NetParameter& param) {
  for(int i = 0; i < param.layers_size(); i++) {
    const LayerParameter& source_layer = param.layers(i);
    const string& source_layer_name = source_layer.name();
    map<string, boost::shared_ptr<Layer>>::iterator layerit = layers_.find(source_layer_name);
    if (layerit == layers_.end()) continue;
    (layerit->second)->copy_trained_layer(source_layer);
  }
  completed_tree_ = false;
  func_tree_.clear();
  return init();
}

bool Pipeline::copy_trained_layers(const string filename) { 
  NetParameter param;
  ReadNetParamsFromBinaryFileOrDie(filename, &param);
  return copy_trained_layers(param);
}

void Pipeline::filter_pipeline(const NetParameter& param,
		     NetParameter *const param_filtered) {
  NetState net_state(param.state());
  if (!net_state.has_phase()) {
    switch (Latte::singleton()->phase()) {
    case Latte::Phase::TRAIN:
      net_state.set_phase(TRAIN);
      break;
    case Latte::Phase::TEST:
      net_state.set_phase(TEST);
      break;
    default:
      LOG(FATAL) << "Unknown phase " << Latte::singleton()->phase();
    }
  }
  param_filtered->CopyFrom(param);
  param_filtered->clear_layers();
  for (int i = 0; i < param.layers_size(); ++i) {
    const LayerParameter& layer_param = param.layers(i);
    CHECK(layer_param.include_size() == 0 || layer_param.exclude_size() == 0)
      << "Specify either include rules or exclude rules; not both.";
    bool layer_included = (layer_param.include_size() == 0);
    for (int j = 0; layer_included && j < layer_param.exclude_size(); ++j) {
      if (state_meets_rule(net_state, layer_param.exclude(j))) {
	layer_included = false;
      }
    }
    for (int j = 0; !layer_included && j < layer_param.include_size(); ++j) {
      if (state_meets_rule(net_state, layer_param.include(j))) {
	layer_included = true;
      }
    }
    if (layer_included) {
      param_filtered->add_layers()->CopyFrom(layer_param);
    }
  }
}

bool Pipeline::state_meets_rule(const NetState& state, 
				const NetStateRule& rule) {
  if (rule.has_phase())
    if (rule.phase() != state.phase())
      return false;
  if (rule.has_min_level())
    if (state.level() < rule.min_level())
      return false;
  if (rule.has_max_level())
    if (state.level() > rule.max_level())
      return false;
  for (int i = 0; i < rule.stage_size(); ++i) {
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (!has_stage) {
      return false;
    }
  }
  for (int i = 0; i < rule.not_stage_size(); ++i) {
    bool has_stage = false;
    for (int j = 0; !has_stage && j < state.stage_size(); ++j) {
      if (rule.not_stage(i) == state.stage(j)) { has_stage = true; }
    }
    if (has_stage) {
      return false;
    }
  }
  return true;
}
