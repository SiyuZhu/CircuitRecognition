#ifndef DATA_TRANSFORMER_HPP
#define DATA_TRANSFORMER_HPP

#include "Halide.h"
#include "caffe.pb.h"
#include "io.hpp"
#include <glog/logging.h>
#include <string>

using namespace Halide;
using namespace latte;
using namespace caffe;
using namespace std;

namespace latte {
  class DataTransformer {
    TransformationParameter transformation_param_;
    BlobProto mean_;

  public:
    DataTransformer();
    explicit DataTransformer(const TransformationParameter& param);

    void transform(const int batch_item_id, const Datum& datum,
		   Buffer& output);
  };
}

#endif
