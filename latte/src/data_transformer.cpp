#include "data_transformer.hpp"

namespace latte {
  DataTransformer::DataTransformer() {}
  
  DataTransformer::DataTransformer(const TransformationParameter& param) {
    transformation_param_ = param;
    if(transformation_param_.has_mean_file()) {
      const string mean_file = transformation_param_.mean_file();
      LOG(INFO) << "Loading mean file from " << mean_file;
      ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &mean_);
    }
  }

  void DataTransformer::transform(const int batch_item_it, const Datum& datum,
				  Buffer& output) {
    const string *data;
    if(datum.has_data()) data = &datum.data();
    float* host_data = ((float *) output.host_ptr());
    for(int c = 0; c < datum.channels(); c++) {
      for(int h = 0; h < datum.height(); h++) {
	for(int w = 0; w < datum.width(); w++) {
	  int data_index = w + h*datum.width() + c*datum.width()*datum.height();
	  int host_index = data_index + batch_item_it*datum.width()*datum.height()
	    *datum.channels();
	  if(!datum.has_data() && (datum.float_data_size() == 0))
	    LOG(FATAL) << "The supplied datum was empty" << endl;
	  float data_elem = (datum.has_data() ? (((uint8_t *) (data))[data_index]) :
			     datum.float_data(data_index));
	  float scale = (transformation_param_.has_scale() ? 
			 transformation_param_.scale() : 1);
	  float mean = (transformation_param_.has_mean_file() ? 
			mean_.data(data_index) : 0);
	  host_data[host_index] = (data_elem - mean) * scale;
	}
      }
    }
  }

}


