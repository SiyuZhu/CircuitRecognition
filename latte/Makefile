CXX=g++
CXXFLAGS=-g -Wall -std=c++0x -I ./include -I ./include/util -I ./src/proto `pkg-config --cflags opencv`
LDFLAGS=`pkg-config --libs opencv`
LDLIBS=-lHalide -lpng -lpthread -ldl -lboost_thread -llmdb -lleveldb -lprotobuf -lglog

SRCDIR:=src
BUILDDIR:=build
TARGET:=./mnist_test

SRCEXT:=cpp
SOURCES:= $(shell find src -type f -name "*.$(SRCEXT)")
OBJECTS:= $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))
PROTODIR:=proto
PROTO:=caffe.pb.o

$(TARGET): $(OBJECTS) $(BUILDDIR)/$(PROTODIR)/$(PROTO)
	@echo " Linking..."
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $(TARGET)

.SECONDEXPANSION:

$(OBJECTS): $$(patsubst $(BUILDDIR)/%,$(SRCDIR)/%, $$(patsubst %.o, %.cpp, $$@))
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(BUILDDIR)/$(PROTODIR)/$(PROTO): $(SRCDIR)/$(PROTODIR)/$(PROTO)
	@mkdir -p $(BUILDDIR)/$(PROTODIR)
	cp -n $(SRCDIR)/$(PROTODIR)/$(PROTO) $(BUILDDIR)/$(PROTODIR)/$(PROTO)

.PHONY: all clean depclean cleanall

all: 
	$(MAKE) $(TARGET)

clean:
	rm -f $(TARGET)

depclean:
	rm -rf $(BUILDDIR)

cleanall:
	$(MAKE) clean depclean