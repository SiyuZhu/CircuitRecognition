#include <stdint.h>
#include "producers.hpp"
#include "io.hpp"
#include <string>
#include <vector>

namespace latte {
  DataProducer::DataProducer(const LayerParameter& param) {
    layer_param_ = param;
    data_transformer_ = DataTransformer(param.transform_param());
    initialized = false;
    init();
  }

  DataProducer::~DataProducer() {
    this->JoinPrefetchThread();
    if(data_.defined())
      data_ = ImageParam();
    if(labels_.defined())
      labels_ = ImageParam();
    if(data_host_ != NULL) {
      delete[] data_host_;
      data_host_ = NULL;
    }
    if(labels_host_ != NULL) {
      delete[] labels_host_;
      labels_host_ = NULL;
    }
    // clean up the database resources
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      break;  // do nothing
    case DataParameter_DB_LMDB:
      mdb_cursor_close(mdb_cursor_);
      mdb_close(mdb_env_, mdb_dbi_);
      mdb_txn_abort(mdb_txn_);
      mdb_env_close(mdb_env_);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
  }

  bool DataProducer::init() {
    if(initialized) return true;

    LOG(INFO) << "Starting to initialize " << this->name() << endl;

    // Initialize DB
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      {
	leveldb::DB* db_temp;
	leveldb::Options options = GetLevelDBOptions();
	options.create_if_missing = false;
	LOG(INFO) << "Opening leveldb " << this->layer_param_.data_param().source();
	leveldb::Status status = 
	  leveldb::DB::Open(options, this->layer_param_.data_param().source(), &db_temp);
	CHECK(status.ok()) << "Failed to open leveldb "
			   << this->layer_param_.data_param().source() << std::endl
			   << status.ToString();
	db_.reset(db_temp);
	iter_.reset(db_->NewIterator(leveldb::ReadOptions()));
	iter_->SeekToFirst();
      }
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
      CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
      CHECK_EQ(mdb_env_open(mdb_env_,
			    this->layer_param_.data_param().source().c_str(),
			    MDB_RDONLY|MDB_NOTLS, 0664), MDB_SUCCESS) << "mdb_env_open failed";
      CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
        << "mdb_txn_begin failed";
      CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
        << "mdb_open failed";
      CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
        << "mdb_cursor_open failed";
      LOG(INFO) << "Opening lmdb " << this->layer_param_.data_param().source();
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
	       MDB_SUCCESS) << "mdb_cursor_get failed";
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }

    /* RANDOM SKIP IS CURRENTLY NOT SUPPORTED
    if (this->layer_param_.data_param().rand_skip()) {
      unsigned int skip = caffe_rng_rand() %
	this->layer_param_.data_param().rand_skip();
      LOG(INFO) << "Skipping first " << skip << " data points.";
      while (skip-- > 0) {
	switch (this->layer_param_.data_param().backend()) {
	case DataParameter_DB_LEVELDB:
	  iter_->Next();
        if (!iter_->Valid()) {
          iter_->SeekToFirst();
        }
        break;
	case DataParameter_DB_LMDB:
        if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
            != MDB_SUCCESS) {
          CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
                   MDB_FIRST), MDB_SUCCESS);
        }
        break;
	default:
	  LOG(FATAL) << "Unknown database backend";
	}
      }
    }
    */
    
    CreatePrefetchThread();
    update_batch();

    LOG(INFO) << "Completed initializing " << this->name() << endl;
    
    initialized = true;
    return initialized;
  }

  void DataProducer::InternalThreadEntry() {
    Datum datum;
    const int batch_size = this->layer_param_.data_param().batch_size();

    for (int item_id = 0; item_id < batch_size; ++item_id) {
      // get a datum
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
	CHECK(iter_);
	CHECK(iter_->Valid());
	datum.ParseFromString(iter_->value().ToString());
	break;
      case DataParameter_DB_LMDB:
	CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
				&mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
	datum.ParseFromArray(mdb_value_.mv_data,
			     mdb_value_.mv_size);
	break;
      default:
	LOG(FATAL) << "Unknown database backend";
      }

      if (!prefetched_data_.defined()) {
	prefetched_data_host_ = new float[datum.width()*datum.height()*
					  datum.channels()*batch_size];
	prefetched_data_ = Buffer(type_of<float>(),
				  datum.width(),
				  datum.height(),
				  datum.channels(),
				  batch_size,
				  (uint8_t *) prefetched_data_host_
				  );
      }

      if (datum.has_label() && !prefetched_labels_.defined()) {
	prefetched_labels_host_ = new int32_t[batch_size];
	prefetched_labels_ = Buffer(type_of<int32_t>(), 1, 1, 1, batch_size,
				    (uint8_t *) prefetched_labels_host_);
      }

      this->data_transformer_.transform(item_id, datum, prefetched_data_);
    
      if(datum.has_label())
	prefetched_labels_host_[item_id] = datum.label();

      // go to the next iter
      switch (this->layer_param_.data_param().backend()) {
      case DataParameter_DB_LEVELDB:
	iter_->Next();
	if (!iter_->Valid()) {
	  // We have reached the end. Restart from the first.
	  DLOG(INFO) << "Restarting data prefetching from start.";
	  iter_->SeekToFirst();
	}
	break;
      case DataParameter_DB_LMDB:
	if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
			   &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
	  // We have reached the end. Restart from the first.
	  DLOG(INFO) << "Restarting data prefetching from start.";
	  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
				  &mdb_value_, MDB_FIRST), MDB_SUCCESS);
	}
	break;
      default:
	LOG(FATAL) << "Unknown database backend";
      }
    }
  }

  bool DataProducer::reset_producer() {
    JoinPrefetchThread();
    switch (this->layer_param_.data_param().backend()) {
    case DataParameter_DB_LEVELDB:
      iter_->SeekToFirst();
      break;
    case DataParameter_DB_LMDB:
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,&mdb_value_, MDB_FIRST)
	       , MDB_SUCCESS);
      break;
    default:
      LOG(FATAL) << "Unknown database backend";
    }
    CreatePrefetchThread();
    update_batch();
    return true;
  }

  bool DataProducer::update_batch() {
    JoinPrefetchThread();
    if(data_.defined()) {
      data_.set(prefetched_data_);
      delete[] data_host_;
      data_host_ = prefetched_data_host_;
    } else {
      data_ = ImageParam(prefetched_data_.type(), 4);
      data_.set(prefetched_data_);
      data_host_ = prefetched_data_host_;
    }
    prefetched_data_ = Buffer();
    if(prefetched_labels_.defined()) {
      if(labels_.defined()) {
	labels_.set(prefetched_labels_);
	if(data_host_ != NULL)
	  delete[] labels_host_;
	labels_host_ = prefetched_labels_host_;
      } else {
	labels_ = ImageParam(prefetched_labels_.type(), 4);
	labels_.set(prefetched_labels_);
	labels_host_ = prefetched_labels_host_;
      }
      prefetched_labels_ = Buffer();
    }
    CreatePrefetchThread();
    return true;
  }
    

  void DataProducer::build_tree(map<string, BoundedFunc>* func_tree) {
    init();
    LOG(INFO) << "Started building " << this->name() << endl;
    // insert image batch
    string data_name = layer_param_.top(0);
    Func data(data_name);
    data(x, y, c, n) = cast<float>(data_(x, y, c, n));
    Buffer data_buf = data_.get();
    array<int, 4> data_dims = {{data_buf.extent(0), data_buf.extent(1),
				data_buf.extent(2), data_buf.extent(3)}};
    BoundedFunc data_bf = make_pair(data, data_dims); 
    func_tree->insert(make_pair(data_name, data_bf));

    LOG(INFO) << data_name << ": {" << data_dims[0] << ", " << data_dims[1]
	      << ", " << data_dims[2] << ", " << data_dims[3] << "}" << endl;

    // insert labels
    if(labels_.defined()) {
      string labels_name = layer_param_.top(1);
      Func labels(labels_name);
      labels(x, y, c, n) = cast<int32_t>(labels_(x, y, c, n));
      Buffer labels_buf = labels_.get();
      array<int, 4> labels_dims = {{labels_buf.extent(0), labels_buf.extent(1),
				    labels_buf.extent(2), labels_buf.extent(3)}};
      BoundedFunc labels_bf = make_pair(labels, labels_dims);
      func_tree->insert(make_pair(labels_name, labels_bf));
      LOG(INFO) << labels_name << ": {" << labels_dims[0] << ", " << labels_dims[1]
		<< ", " << labels_dims[2] << ", " << labels_dims[3] << "}" << endl;
	
    }

    LOG(INFO) << "Completed building " << this->name() << endl;
  }
}
