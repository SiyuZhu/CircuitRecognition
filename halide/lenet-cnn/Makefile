# lib_OBJS are dependencies of cnn, which contains the halide pipeline definition.
# To generate the pipeline object and header files, run cnn with the appropriate flags.
#
# minst_tester is a classifier generated using the aot compilation of cnn using mnist
# parameters. It classifies images from the mnist test idx files.
#
# gates_tester is classifier generated using the aot compilation of cnn uing gates
# parameters. It tests images from the gates test_data directory.

USR=/work/zhang/common/usr
TOOLS=/work/zhang/common/tools
UTILS=../../utils

CXXFLAGS=-Wall -g -std=c++0x -I$(USR)/include
LDFLAGS=-L$(USR)/lib -L$(TOOLS)/libjpeg/jpeg-6b
LDLIBS=-lHalide -lpng -ljpeg -lpthread -ldl -lgflags

lib_OBJS=layers.o common.o buffer_io.o
pipeline_OBJS=mnist_pipeline.o gates_pipeline.o mnist_opencl_pipeline.o
pipeline_H=mnist_pipeline.h gates_pipeline.h mnist_opencl_pipeline.h
main_EXES=cnn mnist_tester boost_benchmark time_benchmark gates_tester mnist_opencl_tester

.SECONDEXPANSION:

$(lib_OBJS): $$(basename $$@).cpp $$(basename $$@).hpp
	g++ $(CXXFLAGS) -c $(basename $@).cpp

layers.o: definitions.hpp

cnn: cnn.cpp cnn.hpp $(UTILS)/read_params.h $(lib_OBJS)
	g++ $(CXXFLAGS) $(LDFLAGS) cnn.cpp $(lib_OBJS) $(LDLIBS) -o cnn

mnist_pipeline.o: cnn
	./cnn --pipeline_name="mnist_pipeline" \
		--param_files="../../data/mnist/layer_params0.dat,../../data/mnist/layer_params1.dat,../../data/mnist/layer_params2.dat,../../data/mnist/layer_params3.dat"

mnist_tester: mnist_pipeline.o mnist_pipeline.h mnist_tester.cpp buffer_io.o $(UTILS)/read_mnist.h $(UTILS)/timer.h
	g++ $(CXXFLAGS) $(LDFLAGS) mnist_tester.cpp mnist_pipeline.o buffer_io.o $(LDLIBS) -o mnist_tester

boost_benchmark: mnist_pipeline.o mnist_pipeline.h boost_benchmark.cpp buffer_io.o $(UTILS)/read_mnist.h
	g++ $(CXXFLAGS) $(LDFLAGS) boost_benchmark.cpp mnist_pipeline.o buffer_io.o $(LDLIBS) -o boost_benchmark

time_benchmark: mnist_pipeline.o mnist_pipeline.h time_benchmark.cpp buffer_io.o $(UTILS)/read_mnist.h $(UTILS)/timer.h
	g++ $(CXXFLAGS) $(LDFLAGS) time_benchmark.cpp mnist_pipeline.o buffer_io.o $(LDLIBS) -o time_benchmark

gates_pipeline.o: cnn
	./cnn --pipeline_name="gates_pipeline" \
		--param_files="../../data/gates/layer_params0.dat,../../data/gates/layer_params1.dat,../../data/gates/layer_params2.dat,../../data/gates/layer_params3.dat"

gates_tester: gates_pipeline.o gates_pipeline.h gates_tester.cpp buffer_io.o $(UTILS)/read_caffetxt.h $(UTILS)/timer.h
	g++ $(CXXFLAGS) $(LDFLAGS) gates_tester.cpp gates_pipeline.o buffer_io.o $(LDLIBS) -o gates_tester

mnist_opencl_pipeline.o: cnn
	./cnn --pipeline_name="mnist_opencl_pipeline" --target="opencl" \
		--param_files="../../data/mnist/layer_params0.dat,../../data/mnist/layer_params1.dat,../../data/mnist/layer_params2.dat,../../data/mnist/layer_params3.dat"

mnist_opencl_tester: mnist_opencl_pipeline.o mnist_opencl_pipeline.h mnist_opencl_tester.cpp buffer_io.o $(UTILS)/read_mnist.h $(UTILS)/timer.h
	g++ $(CXXFLAGS) $(LDFLAGS) mnist_opencl_tester.cpp mnist_opencl_pipeline.o buffer_io.o $(LDLIBS) -o mnist_opencl_tester


.PHONY: all clean gpu

all:
	$(MAKE) $(main_EXES)

clean:
	rm -f $(lib_OBJS) $(pipeline_OBJS) $(pipeline_H) $(main_EXES)
	rm -f *.o

gpu: 
	$(MAKE) buffer_io.o
	g++ -Wall -g -std=c++0x -L/usr/lib/x86_64-linux-gnu/ mnist_opencl_tester.cpp mnist_opencl_pipeline.o -o mnist_opencl_tester