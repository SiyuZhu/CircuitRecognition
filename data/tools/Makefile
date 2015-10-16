USR=/work/zhang/common/usr
CXXFLAGS=-g -Wall -Werror -std=c++0x `pkg-config --cflags opencv`
LDFLAGS=-L$(USR)/lib `pkg-config --libs opencv`
LDLIBS=-lboost_filesystem -lboost_system -lgflags -lglog
lib_OBJS=common.o
main_EXE=allocate_images print_image_stats resize_images center_images rename_images augment_images \
	convert_images_idx


.SECONDEXPANSION:

$(lib_OBJS): $$(basename $$@).cpp $$(basename $$@).hpp
	g++ $(CXXFLAGS) -c $(basename $@).cpp

$(main_EXE): $$@.cpp common.o
	g++ $(CXXFLAGS) $(LDFLAGS) $@.cpp common.o $(LDLIBS) -o $@

.PHONY: clean cleanall all

all:
	$(MAKE) $(main_EXE)

clean: 
	rm -f $(main_EXE) $(addsuffix .o, $(main_EXE))

cleanall:
	rm -f $(lib_OBJS) && $(MAKE) clean
