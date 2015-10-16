#ifndef PRODUCERS_HPP
#define PRODUCERS_HPP

#include "stage.hpp"
#include "data_transformer.hpp"
#include "internal_thread.hpp"
#include "leveldb/db.h"
#include "lmdb.h"

using namespace latte;

namespace latte {
  class Producer : public Stage {
  public:
    virtual ~Producer() {}
    virtual vector<string> producer_names() const;
    
    virtual string name() const = 0;
    virtual bool init() = 0;
    virtual vector<string> output_names() const = 0;
    virtual void build_tree(map<string, BoundedFunc>* func_tree) = 0;
    virtual void to_proto(NetParameter& net) const = 0;
    virtual bool update_batch() = 0;
    virtual bool reset_producer() = 0;
  };
}

namespace latte {
  class InputProducer : public Producer {
    string name_;
    array<int,4> dims_;
    
  public:
    virtual ~InputProducer() {}
    InputProducer(string name, array<int,4> dims);
    InputProducer(string name, int x_in, int y_in, int c_in, int n_in);

    virtual string name() const;
    virtual bool init();
    virtual vector<string> output_names() const;
    virtual void build_tree(map<string, BoundedFunc>* func_tree);

    virtual bool update_batch();
    virtual bool reset_producer();
    virtual void to_proto(NetParameter& net) const;
  };
}

namespace latte {
  class BaseProducer : public Producer {
  protected:
    LayerParameter layer_param_;
    DataTransformer data_transformer_;

  public:
    virtual ~BaseProducer() {}
    static bool is_valid(const LayerParameter& param);
    static boost::shared_ptr<Producer> get_producer(const LayerParameter& param);

    virtual string name() const;
    virtual vector<string> output_names() const;
    virtual void to_proto(NetParameter& net) const;
    virtual void to_proto(LayerParameter *const param) const;

    virtual void build_tree(map<string, BoundedFunc>* func_tree) = 0;
    virtual bool update_batch() = 0;
    virtual bool reset_producer() = 0;
  };
}

namespace latte {
  class BasePrefetchingProducer : 
    public BaseProducer, public InternalThread {
  protected:
    float *prefetched_data_host_;
    int32_t *prefetched_labels_host_;
    Buffer prefetched_data_;
    Buffer prefetched_labels_;

  public:
    virtual ~BasePrefetchingProducer();
    
    virtual void build_tree(map<string, BoundedFunc>* func_tree) = 0;
    virtual bool update_batch() = 0;
    virtual bool reset_producer() = 0;

    virtual void CreatePrefetchThread();
    virtual void JoinPrefetchThread();
  };
}

namespace latte {
  class DataProducer : public BasePrefetchingProducer {
    float *data_host_;
    int32_t *labels_host_;
    ImageParam data_;
    ImageParam labels_;
    bool initialized;

    // LEVELDB
    boost::shared_ptr<leveldb::DB> db_;
    boost::shared_ptr<leveldb::Iterator> iter_;
    // LMDB
    MDB_env* mdb_env_;
    MDB_dbi mdb_dbi_;
    MDB_txn* mdb_txn_;
    MDB_cursor* mdb_cursor_;
    MDB_val mdb_key_, mdb_value_;
    
  public: 
    DataProducer(const LayerParameter& param);
    virtual ~DataProducer();

    virtual bool init();
    virtual void InternalThreadEntry();
    virtual bool reset_producer();
    virtual bool update_batch();
    virtual void build_tree(map<string, BoundedFunc>* func_tree);
  };
}

#endif
