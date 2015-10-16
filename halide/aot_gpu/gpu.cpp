#include "./pipe_gen.h"
#include "../support/buffer_io.h"
#include <stdio.h>
#include <stdlib.h>
//#define __PIPE_DEBUG__

int main(int argc, char *argv[])
{
	//
	// Create data for the images
	//
	uint8_t kernel_data[3*3] = {
		1,1,1,
		1,1,1,
		1,1,1
	};
	uint8_t weight = 9;

	buffer_t inputBuf={0};
	buffer_t kernelBuf={0};
	buffer_t outputBuf={0};

	// Load input image in the buffer
	load_buffer("./IMG.png",inputBuf);
	// Initialize kernel and output buffer
	init_buffer(kernelBuf,3,3,0,kernel_data);
	init_buffer(outputBuf, inputBuf.extent[0]-2, inputBuf.extent[1]-2, inputBuf.extent[2]);
	
	// Warm up GPU
	pipe_gen(&inputBuf, &kernelBuf, weight, &outputBuf);
	#ifdef __PIPE_DEBUG__
	// Run the program several times to get average time
	for (int i = 0; i < 200; i++)
	{
		// Call Halide function
		pipe_gen(&inputBuf, &kernelBuf, weight, &outputBuf);
	}
	#endif

	// Save output image
	save_buffer(outputBuf,"out.png");

	return 0;
}

