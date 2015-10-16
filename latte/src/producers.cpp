#include "producers.hpp"

namespace latte {
  vector<string> Producer::producer_names() const {
    return vector<string>();
  }
}

namespace latte {
  InputProducer::InputProducer(string name, array<int,4> dims) {
    name_ = name;
    dims_ = dims;
    long unsigned int size = 4;
    CHECK_EQ(dims.size(), size) << "Must defined all 4 dimensions in input "
				<< name << endl;
  }
  InputProducer::InputProducer(string name, int x_in, int y_in, int c_in, int n_in) {
    name_ = name;
    array<int,4> temp = {{x_in, y_in, c_in, n_in}};
    dims_ = temp;
  }
  string InputProducer::name() const {return name_;}
  bool InputProducer::init() {return false;}
  vector<string> InputProducer::output_names() const {return vector<string>({name_});}
  void InputProducer::build_tree(map<string, BoundedFunc>* func_tree) {
    LOG(FATAL) << "Cannot build a valid function using an input producer"
	       << endl;
  }
  bool InputProducer::update_batch() {return false;}
  bool InputProducer::reset_producer() {return false;}
  void InputProducer::to_proto(NetParameter& net) const {
    *net.add_input() = name_;
    net.add_input_dim(dims_[0]);
    net.add_input_dim(dims_[1]);
    net.add_input_dim(dims_[2]);
    net.add_input_dim(dims_[3]);
  }
}

namespace latte {
  string BaseProducer::name() const {return layer_param_.name();}
  vector<string> BaseProducer::output_names() const {
    vector<string> outputs;
    for(int i = 0, top = layer_param_.top_size(); i < top; i++)
      outputs.push_back(layer_param_.top(i));
    return outputs;
  }
  void BaseProducer::to_proto (NetParameter& net) const {
    *net.add_layers() = layer_param_;
  }
  void BaseProducer::to_proto(LayerParameter *const param) const {
    param->CopyFrom(layer_param_);
  }
}

namespace latte {
  BasePrefetchingProducer::~BasePrefetchingProducer() {
    this->JoinPrefetchThread();
    if(prefetched_data_.defined())
      prefetched_data_ = Buffer();
    if(prefetched_labels_.defined())
      prefetched_labels_ = Buffer(); 
    if(prefetched_data_host_ != NULL) {
      delete[] prefetched_data_host_; 
      prefetched_data_host_ = NULL;
    }
    if(prefetched_labels_host_ != NULL) {
      delete[] prefetched_labels_host_; 
      prefetched_labels_host_ = NULL;
    }
  }

  void BasePrefetchingProducer::CreatePrefetchThread() {
    CHECK(StartInternalThread()) << "Thread execution failed";
  }
  void BasePrefetchingProducer::JoinPrefetchThread() {
    CHECK(WaitForInternalThreadToExit()) << "Thread joining failed";
  }
}

namespace latte {
  bool BaseProducer::is_valid(const LayerParameter& param) {
    if(!param.has_type()) return false;
    switch (param.type()) {
    case LayerParameter_LayerType_DATA: return true; 
    default: return false;
    }
    return false;
  }
  boost::shared_ptr<Producer> BaseProducer::get_producer(const LayerParameter& param) {
    CHECK(BaseProducer::is_valid(param)) << "Cannot generate data producer\""
					 << param.name() << "\"." << endl;
    switch (param.type()) {
    case LayerParameter_LayerType_DATA: 
      return boost::shared_ptr<Producer>(new DataProducer(param));
      break;
    default: LOG(FATAL) << "Cannot generate data producer\"" << param.name()
			<< "\"."  << endl;
    }
    return boost::shared_ptr<Producer>(); // Dummy return. will never be used.
  }
}
