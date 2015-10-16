HFLAGS=-lHalide -lpng -lpthread -ldl
CFLAGS=-g

HDEF=-D__USE_GPU__ -D__OPENCL__
HCUDADEF=-D__USE_GPU__ -D__CUDA__

#define HDEBUG when calling makefile
#-D__PIPEDEBUG__
HDEBUG=

GPUFLAGS=-lpng -lpthread -ldl -lcuda
GPU_OPENCLFLAGS=-lpng -lpthread -ldl -lOpenCL
CPUFLAGS=-L/work/zhang/common/usr/lib -lpng -lpthread -ldl

TARGET=pipe_gpu
COMPILED=pipe_gen.h pipe_gen.cpp gpu_run cfile.cpp out.png codegen.txt

all: $(TARGET)

$(TARGET): $(TARGET).cpp 
	g++ $(CFLAGS) $^ -o $@ $(HDEF) $(HDEBUG) $(HFLAGS)
	HL_DEBUG_CODEGEN=2 ./$(TARGET) 2> codegen.txt

gen_cuda: $(TARGET).cpp 
	g++ $(CFLAGS) $^ -o $(TARGET) $(HCUDADEF) $(HDEBUG) $(HFLAGS)
	HL_DEBUG_CODEGEN=2 ./$(TARGET) 2> codegen.txt

gen_cpu: $(TARGET).cpp 
	g++ $(CFLAGS) $^ -o $(TARGET) $(HFLAGS) $(HDEBUG)
	HL_DEBUG_CODEGEN=2 ./$(TARGET) 2> codegen.txt

cuda: gpu.cpp pipe_gen.o
	g++ $(CFLAGS) $^ -o gpu_run $(GPUFLAGS)
	time ./gpu_run

opencl: gpu.cpp pipe_gen.o
	g++ $(CFLAGS) $^ -o gpu_run $(GPU_OPENCLFLAGS)
	time ./gpu_run

cpu: gpu.cpp pipe_gen.o
	g++ $(CFLAGS) $^ -o gpu_run $(CPUFLAGS)
	time ./gpu_run

clean:
	rm -f $(TARGET) *.o *.dat $(COMPILED)

