#include <Halide.h>
#include <stdio.h>

using namespace Halide;

int main(int argc, char *argv[])
{
	//
	// Creating input, kernel and output image
	//
	Param<uint8_t> weight("weight");
	ImageParam input_img(UInt(8),3,"input");
	ImageParam kernel_img(UInt(8),2,"kernel");

	//
	// Converting input image to enable calculations
	//
	Func input("input"), kernel("kernel");
	Var x("x"), y("y"), c("c");
	Var xi("xi"), xo("xo"), yi("yi"), yo("yo"), fused("fused");	

	input(x,y,c) = cast<float>(input_img(x,y,c));
	kernel(x,y) = cast<float>(kernel_img(x,y));
	
	//
	// Convolution Function
	//
	Func convolution("convolution");
	RDom r(0,3,0,3,"r");

	convolution(x,y,c) = sum( kernel(2 - r.x, 2 - r.y) * input(x+2 - r.x, y+2 - r.y, c))/weight;
	convolution(x,y,3) = input(x,y,3);	
	//
	// Converting back
	//
	Func output("output");
	output(x,y,c) = cast<uint8_t>(convolution(x,y,c));
	
	//
	// Target settings
	//
	Target target;
	target.os = Target::Linux;
	target.arch = Target::X86;
	target.bits = 64;
	target.set_feature(Target::SSE41);
	#ifdef __OPENCL__
	target.set_feature(Target::OpenCL);
	#elif defined(__CUDA__)
	target.set_feature(Target::CUDA);
	target.set_feature(Target::CUDACapability35);
	#endif
	
	#ifdef __PIPEDEBUG__
	// See which API calls we do
	target.set_feature(Target::Debug);
	#endif
	
	//
	// Schedule
	//
	
	#ifdef __USE_GPU__
	convolution.compute_root().gpu_tile(x,y,8,8);
	#else
	convolution.compute_root().tile(x,y,xo,yo,xi,yi,8,8);
	#endif
	std::vector<Argument> args;
	args.push_back(input_img);
	args.push_back(kernel_img);
	args.push_back(weight);
	output.compile_to_file("pipe_gen",args, target); 

	return 0;
}
