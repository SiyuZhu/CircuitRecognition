USR=/work/zhang/common/usr
CXXFLAGS=-Wall -g -std=c++0x -I$(USR)/include
LDFLAGS=-L$(USR)/lib
LDLIBS=-lHalide -lpng -lpthread -ldl
aot_OUT=Compiled_Pipeline
lib_OBJ=layers.o
main_EXE=cnn main

.SECONDEXPANSION:

layers.o: params.hpp

$(lib_OBJS): $$(basename $$@).cpp
	g++ $(CXXFLAGS) -c $$(basename $$@).cpp

cnn: cnn.cpp $(lib_OBJ)
	g++ $(CXXFLAGS) $(LDFLAGS) cnn.cpp layers.o params.o $(LDLIBS) -o cnn

.PHONY: all clean

all:
	$(MAKE) cnn

clean:
	rm -f $(lib_OBJ) $(addsuffix .o, $(main_EXE)) $(main_EXE) $(aot_OUT).o $(aot_OUT).h