HFLAGS=-lHalide -lpng -lpthread -ldl
CFLAGS=-Wall -g
TARGET=generate_cnn 
RUN=run_cnn
PIPELINE=Compiled_Pipeline
TEST=test_last test_hidden test_conv
BENCHMARK = boost_benchmark time_benchmark

all: $(TARGET) gen $(RUN)
run: $(RUN)
test: $(TEST)


gen: 
	./generate_cnn

# run_cnn creates the CNN executable
$(RUN): main.o $(PIPELINE).o
	g++ $(LDFLAGS) main.o $(PIPELINE).o -o $@ $(HFLAGS)

boost_benchmark: boost_benchmark.cpp $(PIPELINE).o
		g++ $(CFLAGS) $(LDFLAGS) $(HFLAGS) boost_benchmark.cpp $(PIPELINE).o -o boost_benchmark

time_benchmark: time_benchmark.cpp $(PIPELINE).o
		g++ $(CFLAGS) $(LDFLAGS) $(HFLAGS) time_benchmark.cpp $(PIPELINE).o -o time_benchmark

# generate_cnn creates the Halide program to generate the CNN pipeline
$(TARGET): cnn.o convolution.o layers.o params.o
	g++ $(LDFLAGS) cnn.o convolution.o layers.o params.o -o $@ $(HFLAGS)

test_last: test_last.o convolution.o layers.o
	g++ $(LDFLAGS) test_last.o convolution.o layers.o -o $@ $(HFLAGS)

test_hidden: test_hidden.o convolution.o layers.o
	g++ $(LDFLAGS) test_hidden.o convolution.o layers.o -o $@ $(HFLAGS)

test_conv: test_conv.o convolution.o layers.o
	g++ $(LDFLAGS) test_conv.o convolution.o layers.o -o $@ $(HFLAGS)

main.o: main.cpp
	g++ $(CFLAGS) $< -c $(HLFLAGS)

cnn.o: cnn.cpp layers.cpp
	g++ $(CFLAGS) $< -c $(HLFLAGS)

test_last.o: test_last.cpp layers.cpp
	g++ $(CFLAGS) $< -c $(HLFLAGS)

test_hidden.o: test_hidden.cpp layers.cpp
	g++ $(CFLAGS) $< -c $(HLFLAGS)

test_conv.o: test_conv.cpp convolution.cpp
	g++ $(CFLAGS) $< -c $(HLFLAGS)

layers.o: layers.cpp layers.h convolution.h
	g++ $(CFLAGS) $< -c $(HLFLAGS)

params.o: params.cpp params.h
	g++ $(CFLAGS) $< -c $(HLFLAGS)


%.o: %.cpp %.h
	g++ $(CFLAGS) $< -c $(HLFLAGS)

clean:
	rm -f $(TARGET) $(RUN) $(TEST) $(BENCHMARK) $(PIPELINE).* *.o *.dat
