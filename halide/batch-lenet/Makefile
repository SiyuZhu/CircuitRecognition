HALIDE=/usr/local/halide
UTILS=../../utils

CXXFLAGS=-Wall -g -std=c++0x -I$(HALIDE)/include
LDFLAGS=-L$(HALIDE)/bin
LDLIBS=-lHalide -lpthread -ldl -lgflags

EXES=mnist_cnn aot_test aot_time jit_time gpu_time
LIB:=buffer_io.cpp common.cpp layers.cpp
LIBOBJ:=$(patsubst %.cpp, %.o, $(LIB))

mnist_cnn: $(LIBOBJ) mnist_cnn.o
	g++ $(CXXFLAGS) $(LDFLAGS) buffer_io.o common.o layers.o mnist_cnn.o $(LDLIBS) -o mnist_cnn

jit_time: jit_time.o $(LIBOBJ) cnn.hpp $(UTILS)/read_params.h
	g++ $(CXXFLAGS) $(LDFLAGS) buffer_io.o common.o layers.o jit_time.o $(LDLIBS) -o jit_time

mnist_pipeline.h: mnist_cnn
	./mnist_cnn --pipeline_name="mnist_pipeline" \
		--param_files="../../data/mnist/layer_params0.dat,../../data/mnist/layer_params1.dat,../../data/mnist/layer_params2.dat,../../data/mnist/layer_params3.dat"

aot_time.o: mnist_pipeline.h

aot_time: aot_time.o buffer_io.o common.o $(UTILS)/read_mnist.h
	g++ $(CXXFLAGS) $(LDFLAGS) buffer_io.o common.o mnist_pipeline.o aot_time.o $(LDLIBS) -o aot_time

aot_test.o: mnist_pipeline.h

aot_test: aot_test.o buffer_io.o common.o $(UTILS)/read_mnist.h
	g++ $(CXXFLAGS) $(LDFLAGS) buffer_io.o common.o mnist_pipeline.o aot_test.o $(LDLIBS) -o aot_test

mnist_gpu_pipeline.h: mnist_cnn
	./mnist_cnn --pipeline_name="mnist_gpu_pipeline" --target="opencl" \
		--param_files="../../data/mnist/layer_params0.dat,../../data/mnist/layer_params1.dat,../../data/mnist/layer_params2.dat,../../data/mnist/layer_params3.dat"

gpu_time.o: mnist_gpu_pipeline.h

gpu_time: gpu_time.o buffer_io.o common.o $(UTILS)/read_mnist.h
	g++ $(CXXFLAGS) $(LDFLAGS) buffer_io.o common.o mnist_gpu_pipeline.o gpu_time.o $(LDLIBS) -o gpu_time

.PHONY: all clean gpu

all:
	$(MAKE) $(EXES)

clean:
	rm -f mnist_pipeline.h mnist_gpu_pipeline.h
	rm -f $(EXES)
	rm -f *.o;
