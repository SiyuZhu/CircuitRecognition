#include "latte.hpp"

#define MODEL "./examples/mnist/lenet_train_test.prototxt"
#define WEIGHTS "./examples/mnist/lenet_iter_10000.caffemodel"

int main(int argc, char **argv) {
  LOG(INFO) << "Creating pipeline" << endl;
  Pipeline net(MODEL);
  LOG(INFO) << "Copying layer" << endl;
  net.copy_trained_layers(WEIGHTS);

  LOG(INFO) << "Calling init" << endl;
  net.init();

  LOG(INFO) << "Collecting outputs" << endl;
  vector<string> outputs = net.output_names();  
  for(vector<string>::iterator it = outputs.begin(), it_end = outputs.end();
      it != it_end; it++)
    LOG(INFO) << "Output " << *it << endl;
  vector<Buffer> bufs;
  LOG(INFO) << "Collecting buffers of desired dims" << endl;
  for(vector<string>::iterator it = outputs.begin(), it_end = outputs.end();
      it != it_end; it++) {
    array<int,4> dims = net.dims(*it);    
    LOG(INFO) << "Output " << *it << " size {"
	      << dims[0] << ", " << dims[1] << ", "
	      << dims[2] << ", " << dims[3] << "}" << endl;
    bufs.push_back(Buffer(type_of<float>(), dims[0], dims[1],
			  dims[2], dims[3]));
  }
  for(int i = 0, i_end = bufs.size(); i != i_end; i++) {
    LOG(INFO) << "Realizing output " << outputs[i] << endl;
    net.realize(bufs[i], outputs[i]);
    Image<float> image(bufs[i]);
    LOG(INFO) << outputs[i] << endl;
    LOG(INFO) << image(0, 0, 0, 0) << endl;
  }

  return 0;
}

  
  
  
  
