//Define to enable helper fucntions in buffer_io.h
#define __OPENCL__

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <CL/cl.h>
#include "../support/buffer_io.h"
#include <png.h>
#define MAX_SOURCE_SIZE (0x100000)

int main()
{
	buffer_t inputBuf = {0};
	buffer_t outputBuf = {0};
	buffer_t kernelBuf = {0};
	cl_device_id device_id = NULL;
	cl_context context = NULL;
	cl_command_queue command_queue = NULL;
	cl_mem input_devbuf = NULL, output_devbuf = NULL, kernel_devbuf = NULL;
	cl_program program = NULL;
	cl_kernel kernel = NULL;
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_devices;
	cl_uint ret_num_platforms;
	cl_int ret;

	FILE *fp;
	char fileName[] = "./kernel.cl";
	char *source_str;
	size_t source_size;

	/* Load the source code containing the kernel*/
	fp = fopen(fileName, "r");
	if (!fp) {
		fprintf(stderr, "Failed to load kernel.\n");
		exit(1);
	}
	source_str = (char*)malloc(MAX_SOURCE_SIZE);
	source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
	fclose(fp);

	/* Get Platform and Device Info */
	ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
	ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

	//
	// Creating Context
	//

	/* Create OpenCL context */
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
	if (ret != CL_SUCCESS)
		exit(1);
	/* Create Command Queue */
	command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
	if (ret != CL_SUCCESS)
		exit(1);
	/* Create Kernel Program from the source */
	program = clCreateProgramWithSource(context, 1, (const char **)&source_str,
	(const size_t *)&source_size, &ret);

	/* Build Kernel Program */
	ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
	
	//
	// Allocating Buffers
	//
	load_buffer("./IMG.png", inputBuf);
	init_buffer(outputBuf, inputBuf.extent[0]-2, inputBuf.extent[1]-2, inputBuf.extent[2]);
	uint8_t kernel_data[3*3] = {
		1,1,1,
		1,1,1,
		1,1,1
	};
	uint8_t weight = 9;	
	init_buffer(kernelBuf,3,3,0,kernel_data); 

	size_t size;
	/* Create Output Buffer */
	size = 2968*1942*4*sizeof(uint8_t);
	output_devbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
	/* Create Kernel Buffer */
	size = 3*3*1*sizeof(uint8_t);
	kernel_devbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);
	/* Create Input Buffer */
	size = 2970*1944*4*sizeof(uint8_t);
	input_devbuf = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &ret);


	//
	// Copy Buffers to Device Memory
	//
	opencl_copy_to_device(&inputBuf, input_devbuf, command_queue);
	opencl_copy_to_device(&kernelBuf, kernel_devbuf, command_queue);

	//
	// Create Kernels
	//
	
	int blocksX=371, blocksY=243, blocksZ=1;
	int threadsX=8, threadsY=8, threadsZ=1;
	
	size_t global_dim[3] = {blocksX*threadsX,  blocksY*threadsY,  blocksZ*threadsZ};
	size_t local_dim[3] = {threadsX, threadsY, threadsZ};

	for(int c = 0; c < inputBuf.extent[2]; c++){
		/* Create OpenCL Kernel */
		kernel = clCreateKernel(program, "kernel_convolution_s0_y___block_id_y", &ret);
		/* Set OpenCL Kernel Parameters */
		ret = clSetKernelArg(kernel/*_convolution_s0_c*/ , 0, sizeof(int), (void *)&c);
		ret = clSetKernelArg(kernel/*_convolution_x_extent_realized*/, 1, sizeof(int), (void *)&outputBuf.extent[0]);
		ret = clSetKernelArg(kernel/*_convolution_x_min_realized*/, 2, sizeof(int), (void *)&outputBuf.min[0]);
		ret = clSetKernelArg(kernel/*_convolution_y_extent_realized*/, 3, sizeof(int), (void *)&outputBuf.extent[1]);
		ret = clSetKernelArg(kernel/*_convolution_y_min_realized*/, 4, sizeof(int), (void *)&outputBuf.min[1]);
		ret = clSetKernelArg(kernel/*_input_min_0*/, 5, sizeof(int), (void *)&inputBuf.min[0]);
		ret = clSetKernelArg(kernel/*_input_min_1*/, 6, sizeof(int), (void *)&inputBuf.min[1]);
		ret = clSetKernelArg(kernel/*_input_min_2*/, 7, sizeof(int), (void *)&inputBuf.min[2]);
		ret = clSetKernelArg(kernel/*_input_min_2_required*/, 8, sizeof(int), (void *)&inputBuf.min[2]);
		ret = clSetKernelArg(kernel/*_input_stride_1*/, 9, sizeof(int), (void *)&inputBuf.stride[1]);
		ret = clSetKernelArg(kernel/*_input_stride_2*/, 10, sizeof(int), (void *)&inputBuf.stride[2]);
		ret = clSetKernelArg(kernel/*_kernel_min_0*/, 11, sizeof(int), (void *)&kernelBuf.min[0]);
		ret = clSetKernelArg(kernel/*_kernel_min_1*/, 12, sizeof(int), (void *)&kernelBuf.min[1]);
		ret = clSetKernelArg(kernel/*_kernel_stride_1*/, 13, sizeof(int), (void *)&kernelBuf.stride[1]);
		ret = clSetKernelArg(kernel/*_output_extent_0*/, 14, sizeof(int), (void *)&outputBuf.stride[1]);
		ret = clSetKernelArg(kernel/*_output_extent_1*/, 15, sizeof(int), (void *)&outputBuf.stride[2]);
		ret = clSetKernelArg(kernel/*_output_min_0*/, 16, sizeof(int), (void *)&outputBuf.min[0]);
		ret = clSetKernelArg(kernel/*_output_min_1*/, 17, sizeof(int), (void *)&outputBuf.min[1]);
		ret = clSetKernelArg(kernel/*_weight*/, 18, sizeof(char), (void *)&weight);
		ret = clSetKernelArg(kernel/*&_convolution*/, 19, sizeof(cl_mem), (void *)&output_devbuf);
		ret = clSetKernelArg(kernel/*&_input*/, 20, sizeof(cl_mem), (void *)&input_devbuf);
		ret = clSetKernelArg(kernel/*&_kernel*/, 21, sizeof(cl_mem), (void *)&kernel_devbuf);
		ret = clSetKernelArg(kernel/*&_shared*/, 22, 0, NULL);
		
		/* Execute OpenCL Kernel */
		ret = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL, global_dim, local_dim, 0, NULL, NULL);
		if (ret != CL_SUCCESS) {
			return ret;
		}

		clReleaseKernel(kernel);
	}
	/* Copy results from the memory buffer */
	opencl_copy_to_host(&outputBuf, output_devbuf, command_queue);	
	
	/* Display Result */
	save_buffer(outputBuf, "out.png");

	/* Finalization */
	ret = clFlush(command_queue);
	ret = clFinish(command_queue);
	
	ret = clReleaseMemObject(output_devbuf);
	ret = clReleaseMemObject(input_devbuf);
	ret = clReleaseMemObject(kernel_devbuf);
	
	ret = clReleaseProgram(program);
	ret = clReleaseCommandQueue(command_queue);
	ret = clReleaseContext(context);

	free(source_str);

	return 0;
}
